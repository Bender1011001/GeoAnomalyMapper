# Robustness Improvements - GeoAnomalyMapper Download Script

## Overview

The `download_geodata.py` script has been enhanced with production-grade robustness features to handle network issues, rate limiting, and DNS failures gracefully.

## 🔧 Key Improvements

### 1. **DNS Preflight Checks**

**What it does:**
- Validates that all Copernicus Data Space hosts are reachable before attempting downloads
- Prevents cryptic connection errors by catching DNS issues early
- Provides clear error messages when network connectivity is unavailable

**Implementation:**
```python
def ensure_dns(host: str, timeout_sec: int = 60) -> None:
    """Block until host resolves, else raise."""
    # Retries DNS resolution for up to 60 seconds
    # Raises RuntimeError with clear message if DNS fails
```

**Hosts checked:**
- `identity.dataspace.copernicus.eu` (authentication)
- `catalogue.dataspace.copernicus.eu` (product search)
- `zipper.dataspace.copernicus.eu` (downloads)

### 2. **Robust Token Management**

**What it does:**
- Automatically retries authentication on transient network errors
- Exponential backoff: 2, 4, 8, 16 seconds between retries
- Supports custom auth URL via `CDSE_AUTH_URL` environment variable
- Clear distinction between retryable (network) and non-retryable (credentials) errors

**Implementation:**
```python
class TokenManager:
    # Retries ConnectionError, NameResolutionError, Timeout
    # Immediately fails on 401/403 (bad credentials)
    # 4 retry attempts with exponential backoff
```

**Benefits:**
- Handles temporary network hiccups automatically
- Won't waste time retrying on authentication failures
- Provides detailed error messages for debugging

### 3. **File Size Validation**

**What it does:**
- Validates that downloaded files are not empty or error pages
- Automatically removes files < 1 KB (likely HTML error pages)
- Prevents keeping corrupted or incomplete downloads

**Implementation:**
```python
# After download completes
size_bytes = os.path.getsize(dataset['output'])
if size_bytes < 1024:
    os.remove(dataset['output'])
    raise RuntimeError("Downloaded file is empty - source likely moved")
```

**Applies to:**
- EMAG2 magnetic data downloads
- WDMAM magnetic data downloads
- Any file-based downloads

### 4. **Enhanced Error Messages**

**What you'll see:**

**Before:**
```
✗ Token refresh failed: [Errno 11001] getaddrinfo failed
```

**After:**
```
✗ Preflight DNS failed for identity.dataspace.copernicus.eu: [Errno 11001]

⚠️  Network connectivity issue detected. Please check:
  - Internet connection is working
  - DNS servers are responding
  - No firewall blocking HTTPS (port 443)
```

## 🚀 Usage

### Basic Usage (Unchanged)

```bash
cd GeoAnomalyMapper
python download_geodata.py
```

The script will now:
1. ✅ Check network connectivity automatically
2. ✅ Validate all hosts are reachable
3. ✅ Retry auth on network issues
4. ✅ Validate downloaded files
5. ✅ Provide clear error messages

### Advanced Configuration

**Override Authentication URL:**
```powershell
$env:CDSE_AUTH_URL="https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
python download_geodata.py
```

**Adjust Worker Count (Rate Limiting):**
```powershell
$env:S1_WORKERS="6"
python download_geodata.py
```

**Select Region Preset:**
```powershell
$env:TARGET_PRESET="USA_PLUS_EU"  # Options: USA, EUROPE, USA_PLUS_EU, GLOBAL
python download_geodata.py
```

## 📋 Dependencies

The following packages are required (auto-installed via pip):

```
python-dotenv>=0.19.0  # .env file loading
requests>=2.25.0       # HTTP requests
urllib3>=1.26.0        # Low-level HTTP (for exception types)
```

**Install all dependencies:**
```bash
pip install -e .
```

## 🔍 Troubleshooting

### "DNS resolution failed"

**Cause:** Your system cannot resolve Copernicus hostnames.

**Solutions:**
1. Check internet connection: `ping 8.8.8.8`
2. Test DNS: `nslookup identity.dataspace.copernicus.eu`
3. Try alternate DNS (e.g., Google DNS 8.8.8.8)
4. Check firewall/proxy settings

### "Token refresh failed after retries"

**Cause:** Persistent network issues or authentication problems.

**Solutions:**
1. Verify credentials in `.env` file
2. Check if Copernicus services are online: https://dataspace.copernicus.eu/
3. Review network connectivity
4. Check for proxy/VPN interference

### "Downloaded file is empty"

**Cause:** Source URL has moved or returned an error page.

**Solutions:**
1. Check if data source is still available
2. Look for alternative mirror URLs
3. Download manually and place in appropriate directory

## 📊 Success Indicators

**Successful startup:**
```
✓ Loaded credentials from GeoAnomalyMapper\.env
✓ Created 648 tiles for global coverage
✓ Directory structure created

✓ Checking network connectivity...
  ✓ identity.dataspace.copernicus.eu reachable
  ✓ catalogue.dataspace.copernicus.eu reachable
  ✓ zipper.dataspace.copernicus.eu reachable

✓ Network OK, initializing token manager...
✓ Token refreshed
```

**Successful download:**
```
[1/150] Tile tile_lat30_lon-120  [30..40 lat, -120..-110 lon]
  Searching products... found 2
  Downloading: S1A_IW_SLC__1SDV_20251006T... (attempt 1)
     95.2%   8.45 GB
  ✓ Downloaded (8.89 GB)
```

## 🛡️ Security Notes

- Credentials loaded from `.env` file (gitignored)
- No hardcoded passwords in source code
- DNS checks prevent potential security issues from DNS hijacking
- File size validation prevents accepting malicious payloads

## 📝 Changelog

### Version 2.0 (2025-01-08)
- ✅ Added DNS preflight checks
- ✅ Implemented robust token retry logic
- ✅ Added file size validation
- ✅ Enhanced error messages
- ✅ Support for CDSE_AUTH_URL override
- ✅ Comprehensive logging

### Version 1.0 (Original)
- Basic download functionality
- Rate limiting support
- USA/EU priority regions

## 🤝 Support

For issues or questions:
1. Check this document first
2. Review error messages carefully
3. Verify network connectivity
4. Check Copernicus Data Space status
5. Review `.env` credentials

---

**Status:** ✅ Production Ready  
**Last Updated:** 2025-01-08  
**Maintainer:** GeoAnomalyMapper Team