# Windows Quick Start Guide

This guide shows the correct syntax for running commands on **Windows PowerShell or Command Prompt**.

## Key Difference

**Unix/Linux/Mac uses backslash `\` for line continuation:**
```bash
python script.py \
  --option1 value1 \
  --option2 value2
```

**Windows PowerShell uses backtick `` ` `` for line continuation:**
```powershell
python script.py `
  --option1 value1 `
  --option2 value2
```

**Best for Windows: Use one line (works in PowerShell AND Command Prompt):**
```powershell
python script.py --option1 value1 --option2 value2
```

---

## Quick Examples for Windows

### Example 1: Basic Fusion (100m resolution)

```powershell
# NEW PowerShell-friendly syntax (NO COMMAS!):
python multi_resolution_fusion.py --lon-min -105.0 --lat-min 32.0 --lon-max -104.0 --lat-max 33.0 --resolution 0.001 --output my_fusion

# Or use defaults (Carlsbad Caverns area):
python multi_resolution_fusion.py --output my_fusion
```

### Example 2: High-Resolution Fusion (10m with InSAR)

```powershell
# Specify custom region:
python multi_resolution_fusion.py --lon-min -105.0 --lat-min 32.0 --lon-max -104.0 --lat-max 33.0 --resolution 0.0001 --output ultra_hires
```

### Example 3: Void Detection

```powershell
# Void detection (still uses comma format - will be updated):
python detect_voids.py --region '-104.5,32.1,-104.0,32.3' --resolution 0.0001 --output carlsbad_voids
```

### Example 4: Download Sentinel-1 Data

```powershell
python download_geodata.py
# Follow the interactive prompts
```

---

## Complete Workflow (Windows)

```powershell
# 1. Set up environment
pip install rasterio scipy numpy matplotlib tqdm requests python-dotenv

# 2. Configure credentials
copy .env.example .env
# Edit .env with your Copernicus credentials

# 3. Download high-resolution data
python download_geodata.py

# 4. Run multi-resolution fusion (PowerShell-friendly syntax!)
python multi_resolution_fusion.py --lon-min -105.0 --lat-min 32.0 --lon-max -104.0 --lat-max 33.0 --resolution 0.001 --output my_map

# 5. Detect voids
python detect_voids.py --region '-105.0,32.0,-104.0,33.0' --resolution 0.001 --output my_voids
```

---

## Troubleshooting Windows Issues

### "python not recognized"

**Solution:** Use `python` or `py`:
```powershell
py -m pip install rasterio
py multi_resolution_fusion.py --lon-min -105.0 --lat-min 32.0 --lon-max -104.0 --lat-max 33.0 --resolution 0.001 --output test
```

### "Cannot load module"

**Solution:** Activate virtual environment first:
```powershell
.venv\Scripts\Activate.ps1
python multi_resolution_fusion.py --lon-min -105.0 --lat-min 32.0 --lon-max -104.0 --lat-max 33.0 --resolution 0.001 --output test
```

### PowerShell Execution Policy Error

**Solution:** Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Path Issues with Data Files

**NEW: PowerShell-Friendly Syntax (No Commas!)**
```powershell
# BEST: Use separate coordinate arguments (works on all platforms):
python multi_resolution_fusion.py --lon-min -105.0 --lat-min 32.0 --lon-max -104.0 --lat-max 33.0 --resolution 0.001 --output test

# Or use defaults:
python multi_resolution_fusion.py --output test

# Paths will be: data/outputs/multi_resolution/test.tif (Python converts automatically)
```

---

## File Paths on Windows

The scripts automatically handle Windows paths. You don't need to change anything in the code:

- **Input data:** `data\raw\gravity\...` (Windows) → converted internally
- **Output:** `data\outputs\multi_resolution\...` (Windows) → works automatically

---

## Performance Tips for Windows

### Use More Workers for Downloads

Edit `download_geodata.py` line where workers are defined, or when prompted choose 8-10 workers:
```
Parallel workers [1-10, default 2]: 8
```

### Check Available RAM

```powershell
# Check available memory
systeminfo | findstr /C:"Available Physical Memory"

# For high-resolution (0.0001°), you need at least 8GB free
# For medium (0.001°), 2-4GB is sufficient
```

### Use SSD for Data Storage

Store `data/` folder on SSD for faster processing:
```powershell
# Move data to SSD if needed
move data D:\SSD\SAR-project\data
# Update paths in scripts or use symbolic link
```

---

## Next Steps

Once commands run successfully:
1. **View results** in `data\outputs\multi_resolution\`
2. **Read full guides:**
   - [QUICKSTART_HIRES.md](QUICKSTART_HIRES.md)
   - [HIGH_RESOLUTION_DATA_GUIDE.md](HIGH_RESOLUTION_DATA_GUIDE.md)
3. **Experiment** with different regions and resolutions

---

## Still Having Issues?

Check the main [README.md](README.md) or open an issue on GitHub with:
- Your exact command
- Full error message
- Output of `python --version`
- Output of `pip list | findstr rasterio`