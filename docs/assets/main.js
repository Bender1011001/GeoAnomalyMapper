const map = L.map('map-canvas', {
  zoomControl: false,
  worldCopyJump: true,
});

const basemap = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution:
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  minZoom: 2,
  maxZoom: 19,
});

basemap.addTo(map);
L.control.zoom({ position: 'bottomright' }).addTo(map);

const layerGroup = L.layerGroup().addTo(map);
const layerIndex = new Map();

const datasetList = document.getElementById('dataset-list');
const legend = document.getElementById('layer-legend');

function setMapBounds() {
  if (layerGroup.getLayers().length === 0) {
    map.setView([0, 0], 2);
    return;
  }

  const bounds = layerGroup.getBounds();
  if (bounds.isValid()) {
    map.fitBounds(bounds.pad(0.1));
  }
}

function createDatasetCard(dataset, featureCount) {
  const card = document.createElement('article');
  card.className = 'dataset-card';
  card.innerHTML = `
    <h3>${dataset.name}</h3>
    <p>${dataset.description || 'No description provided.'}</p>
    <div class="dataset-meta">
      <div class="dataset-color"><span style="background:${dataset.color || '#ff5555'}"></span>${
    dataset.color || '#ff5555'
  }</div>
      <div>Features: ${featureCount ?? '–'}</div>
      <div>Source: <code>${dataset.file}</code></div>
    </div>
    <label class="toggle" data-layer-id="${dataset.id}">
      <input type="checkbox" checked />
      <span>Show on map</span>
    </label>
  `;

  const checkbox = card.querySelector('input[type="checkbox"]');
  checkbox.addEventListener('change', (event) => {
    const layer = layerIndex.get(dataset.id);
    if (!layer) return;
    if (event.target.checked) {
      layerGroup.addLayer(layer);
    } else {
      layerGroup.removeLayer(layer);
    }
    setMapBounds();
  });

  datasetList.appendChild(card);
}

function addLegendEntry(dataset) {
  const item = document.createElement('div');
  item.className = 'legend-item';
  item.innerHTML = `
    <span class="legend-swatch" style="background:${dataset.color || '#ff5555'}"></span>
    <span>${dataset.name}</span>
  `;
  legend.appendChild(item);
}

function drawDataset(dataset, geojson) {
  const layer = L.geoJSON(geojson, {
    style: () => ({
      color: dataset.color || '#ff5555',
      weight: 2,
      opacity: 0.9,
      fillOpacity: 0.15,
    }),
    pointToLayer: (feature, latlng) =>
      L.circleMarker(latlng, {
        radius: 6,
        color: dataset.color || '#ff5555',
        weight: 2,
        fillColor: dataset.color || '#ff5555',
        fillOpacity: 0.6,
      }),
    onEachFeature: (feature, layer) => {
      const props = feature.properties || {};
      const lines = Object.entries(props).map(([key, value]) => `<strong>${key}</strong>: ${value}`);
      layer.bindPopup(
        `
          <div class="popup">
            <h4>${dataset.name}</h4>
            ${lines.length ? `<div class="props">${lines.join('<br/>')}</div>` : '<em>No attributes</em>'}
          </div>
        `
      );
    },
  });

  layerIndex.set(dataset.id, layer);
  layerGroup.addLayer(layer);
  addLegendEntry(dataset);
  createDatasetCard(dataset, geojson.features?.length ?? null);
  setMapBounds();
}

function reportError(message) {
  const card = document.createElement('article');
  card.className = 'dataset-card';
  card.innerHTML = `<h3>Dataset error</h3><p>${message}</p>`;
  datasetList.appendChild(card);
}

async function loadDatasets() {
  try {
    const response = await fetch('data/datasets.json', { cache: 'no-store' });
    if (!response.ok) {
      if (response.status === 404) {
        reportError(
          'No datasets configuration found. Add a data/datasets.json file to publish processed results.'
        );
      }
      return;
    }
    const datasets = await response.json();
    if (!Array.isArray(datasets) || datasets.length === 0) {
      reportError('datasets.json is empty. Add at least one dataset to visualise it on the map.');
      return;
    }

    await Promise.all(
      datasets.map(async (dataset) => {
        try {
          const geoResponse = await fetch(dataset.file, { cache: 'no-store' });
          if (!geoResponse.ok) {
            reportError(`Could not load ${dataset.file}. Verify the path and try again.`);
            return;
          }
          const geojson = await geoResponse.json();
          drawDataset(dataset, geojson);
        } catch (error) {
          reportError(`Failed to render ${dataset.name}: ${error.message}`);
        }
      })
    );
  } catch (error) {
    reportError(`Unable to read datasets.json: ${error.message}`);
  }
}

async function handleFileUpload(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  try {
    const text = await file.text();
    const geojson = JSON.parse(text);
    const dataset = {
      id: `local-${Date.now()}`,
      name: file.name,
      color: '#ffa500',
      description: 'Local preview only',
      file: file.name,
    };

    drawDataset(dataset, geojson);
  } catch (error) {
    alert(`Could not parse GeoJSON file: ${error.message}`);
  }
}

document.getElementById('file-input').addEventListener('change', handleFileUpload);

loadDatasets();
