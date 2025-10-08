# Security Policy

## Credential Management

### ⚠️ CRITICAL: Never Commit Credentials

This project requires Copernicus Data Space credentials to download Sentinel-1 InSAR data. **Never commit credentials to version control.**

### Proper Setup

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Add your credentials to `.env`:**
   ```bash
   CDSE_USERNAME=your_email@example.com
   CDSE_PASSWORD=your_password
   ```

3. **Verify `.env` is in `.gitignore`:**
   ```bash
   git check-ignore .env
   # Should output: .env
   ```

### Registration

Get free credentials at: https://dataspace.copernicus.eu/

1. Click "Register"
2. Verify your email
3. Use credentials in `.env` file

### Environment Variable Loading

The script automatically loads credentials from environment variables:
- `CDSE_USERNAME` - Your Copernicus username (email)
- `CDSE_PASSWORD` - Your Copernicus password

You can also set these in your shell:
```bash
export CDSE_USERNAME="your_email@example.com"
export CDSE_PASSWORD="your_password"
python download_geodata.py
```

### Security Best Practices

✅ **DO:**
- Use `.env` file for local development
- Keep `.env.example` updated with required variables (no values!)
- Use environment variables in CI/CD pipelines
- Rotate credentials regularly
- Use different credentials for dev/prod

❌ **DON'T:**
- Hardcode credentials in Python files
- Commit `.env` to git
- Share credentials in chat/email
- Use production credentials in development
- Log credentials in debug output

## Reporting Security Issues

If you discover a security vulnerability, please email the maintainer directly rather than using the issue tracker.

## Recent Security Fixes

### 2025-10-08: Removed Hardcoded Credentials
- **Issue:** Credentials were hardcoded as fallback values in `download_geodata.py`
- **Fix:** Removed all hardcoded credentials; script now requires environment variables
- **Impact:** Prevents accidental credential exposure in version control