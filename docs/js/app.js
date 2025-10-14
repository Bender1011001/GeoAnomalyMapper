const map = L.map('map-view', {
  zoomSnap: 0.25,
  scrollWheelZoom: true,
});

const baseLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; <a href="https://openstreetmap.org">OpenStreetMap</a> contributors',
});

baseLayer.addTo(map);

const detectionsLayer = L.geoJSON(null, {
  pointToLayer: (feature, latlng) => {
    const confidence = feature.properties?.confidence ?? 0;
    const radius = Math.max(6, Math.min(22, (feature.properties?.radius_m ?? 15) / 5));

    const marker = L.circleMarker(latlng, {
      radius,
      color: '#1e86d9',
      weight: 2,
      opacity: 0.9,
      fillColor: confidence >= 0.75 ? '#0abf9b' : '#44c0ff',
      fillOpacity: 0.7,
    });

    marker.bindPopup(createPopupContent(feature));
    marker.on('popupopen', () => highlightCard(feature.properties.__uiId));
    return marker;
  },
});

detectionsLayer.addTo(map);

let heatLayer = null;
let detections = [];

const detectionList = document.getElementById('detection-list');
const confidenceFilter = document.getElementById('confidence-filter');
const confidenceLabel = document.getElementById('confidence-label');
const showHeatmap = document.getElementById('show-heatmap');

const statTotal = document.getElementById('stat-total');
const statConfidence = document.getElementById('stat-confidence');
const statDataset = document.getElementById('stat-dataset');

async function init() {
  setCurrentYear();
  confidenceLabel.textContent = Number(confidenceFilter.value).toFixed(2);

  try {
    const response = await fetch('data/voids.geojson', { cache: 'no-cache' });
    if (!response.ok) {
      throw new Error(`Failed to load data: ${response.status} ${response.statusText}`);
    }
    const data = await response.json();
    prepareData(data);
    updateView();
  } catch (error) {
    console.error(error);
    displayError(`Unable to load detections. Add a GeoJSON export to <code>docs/data/voids.geojson</code> and redeploy.`);
  }
}

function prepareData(data) {
  if (!data || !Array.isArray(data.features)) {
    throw new Error('GeoJSON FeatureCollection expected.');
  }

  detections = data.features
    .map((feature, index) => ({
      ...feature,
      properties: {
        ...feature.properties,
        __uiId: `detection-${index}`,
      },
    }))
    .sort((a, b) => (b.properties?.confidence ?? 0) - (a.properties?.confidence ?? 0));

  updateStats(detections);
}

function updateStats(features) {
  const total = features.length;
  statTotal.textContent = total ? total.toLocaleString() : '0';

  if (!total) {
    statConfidence.textContent = '—';
    statDataset.textContent = '—';
    return;
  }

  const averageConfidence =
    features.reduce((acc, feature) => acc + (feature.properties?.confidence ?? 0), 0) /
    total;
  statConfidence.textContent = `${averageConfidence.toFixed(2)}`;

  const newestDataset = features
    .map((feature) => ({
      dataset: feature.properties?.dataset ?? 'Unknown dataset',
      time: feature.properties?.timestamp ?? feature.properties?.date ?? '',
    }))
    .sort((a, b) => (b.time || '').localeCompare(a.time || ''))[0];

  statDataset.textContent = newestDataset?.dataset ?? 'Unknown dataset';
}

function updateView() {
  const minConfidence = Number(confidenceFilter.value);
  confidenceLabel.textContent = minConfidence.toFixed(2);

  const filtered = detections.filter(
    (feature) => (feature.properties?.confidence ?? 0) >= minConfidence,
  );

  detectionsLayer.clearLayers();
  detectionsLayer.addData(filtered);

  updateHeatmap(filtered);
  renderDetectionList(filtered);

  if (filtered.length) {
    const bounds = detectionsLayer.getBounds();
    if (bounds.isValid()) {
      map.fitBounds(bounds.pad(0.15));
    }
  }
}

function renderDetectionList(features) {
  detectionList.innerHTML = '';
  if (!features.length) {
    const emptyState = document.createElement('li');
    emptyState.className = 'detection';
    emptyState.innerHTML =
      '<h4>No detections match the current filters.</h4><p>Lower the confidence threshold or upload new data.</p>';
    detectionList.appendChild(emptyState);
    return;
  }

  features.forEach((feature) => {
    const properties = feature.properties ?? {};
    const item = document.createElement('li');
    item.className = 'detection';
    item.dataset.id = properties.__uiId;

    const confidence = properties.confidence ?? null;
    const depth = properties.depth_m ?? null;
    const radius = properties.radius_m ?? null;

    item.innerHTML = `
      <h4>${properties.name || properties.id || 'Unnamed detection'}</h4>
      <div class="detection__meta">
        <span title="Confidence score">⚡ ${formatConfidence(confidence)}</span>
        <span title="Estimated depth">🕳️ ${formatDepth(depth)}</span>
        <span title="Estimated radius">📏 ${formatRadius(radius)}</span>
        <span title="Source dataset">🗂️ ${properties.dataset || 'Unknown'}</span>
      </div>
      ${properties.notes ? `<p>${properties.notes}</p>` : ''}
    `;

    item.addEventListener('click', () => {
      highlightCard(properties.__uiId);
      openPopupForFeature(feature);
    });

    detectionList.appendChild(item);
  });
}

function highlightCard(id) {
  detectionList.querySelectorAll('.detection').forEach((card) => {
    card.classList.toggle('detection--active', card.dataset.id === id);
  });
}

function openPopupForFeature(feature) {
  detectionsLayer.eachLayer((layer) => {
    if (layer.feature?.properties?.__uiId === feature.properties.__uiId) {
      layer.openPopup();
      map.panTo(layer.getLatLng());
    }
  });
}

function updateHeatmap(features) {
  if (heatLayer) {
    map.removeLayer(heatLayer);
    heatLayer = null;
  }

  if (!showHeatmap.checked || !features.length) {
    return;
  }

  const heatPoints = features
    .map((feature) => {
      const coords = feature.geometry?.coordinates;
      if (!Array.isArray(coords) || coords.length < 2) return null;
      const [lon, lat] = coords;
      const intensity = Math.max(0.15, Math.min(1, feature.properties?.confidence ?? 0.5));
      return [lat, lon, intensity];
    })
    .filter(Boolean);

  if (!heatPoints.length) return;

  heatLayer = L.heatLayer(heatPoints, {
    radius: 28,
    blur: 32,
    minOpacity: 0.4,
  });
  heatLayer.addTo(map);
}

function createPopupContent(feature) {
  const properties = feature.properties ?? {};
  const title = properties.name || properties.id || 'Detection';

  const details = [
    `<strong>Confidence:</strong> ${formatConfidence(properties.confidence)}`,
    `<strong>Depth:</strong> ${formatDepth(properties.depth_m)}`,
    `<strong>Radius:</strong> ${formatRadius(properties.radius_m)}`,
    `<strong>Dataset:</strong> ${properties.dataset || 'Unknown dataset'}`,
  ];

  if (properties.notes) {
    details.push(`<strong>Notes:</strong> ${properties.notes}`);
  }

  const additional = Object.entries(properties)
    .filter(([key]) => !['id', 'name', 'confidence', 'depth_m', 'radius_m', 'dataset', 'notes', '__uiId'].includes(key))
    .map(([key, value]) => `<strong>${key}:</strong> ${value}`);

  return `
    <div class="popup">
      <h4>${title}</h4>
      <div class="popup__details">${[...details, ...additional].join('<br />')}</div>
    </div>
  `;
}

function formatConfidence(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'n/a';
  }
  return `${(value * 100).toFixed(0)}%`;
}

function formatDepth(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'n/a';
  return `${value.toLocaleString()} m`;
}

function formatRadius(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'n/a';
  return `${value.toLocaleString()} m`;
}

function displayError(message) {
  const section = document.querySelector('#map .map-layout');
  section.innerHTML = `
    <div class="error">
      <h3>Data missing</h3>
      <p>${message}</p>
      <p>
        Need an example? Review <code>docs/data/sample_voids.geojson</code> for the
        expected format.
      </p>
    </div>
  `;
}

function setCurrentYear() {
  const yearEl = document.getElementById('year');
  if (yearEl) {
    yearEl.textContent = new Date().getFullYear();
  }
}

confidenceFilter.addEventListener('input', () => updateView());
showHeatmap.addEventListener('change', () => updateView());

document.addEventListener('DOMContentLoaded', init);
