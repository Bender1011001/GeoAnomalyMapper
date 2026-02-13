"""
GeoAnomalyMapper Showcase PDF Generator
Converts the HTML showcase to a professional PDF document (under 2MB)

Requirements:
    pip install playwright pypdf
    playwright install chromium
"""

import asyncio
import sys
from pathlib import Path

async def generate_pdf():
    """Generate a high-quality PDF from the HTML showcase."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("❌ Playwright not installed. Installing now...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
        from playwright.async_api import async_playwright
    
    script_dir = Path(__file__).parent
    html_path = script_dir / "GeoAnomalyMapper_Showcase.html"
    pdf_path = script_dir / "GeoAnomalyMapper_Portfolio_Showcase.pdf"
    
    if not html_path.exists():
        print(f"❌ HTML file not found: {html_path}")
        return False
    
    print("🚀 Starting PDF generation...")
    print(f"   📄 Source: {html_path}")
    print(f"   📑 Output: {pdf_path}")
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Load the HTML file
        await page.goto(f"file:///{html_path.absolute()}")
        
        # Wait for fonts and images to load
        await page.wait_for_timeout(2000)
        
        # Generate PDF with optimized settings for smaller file size
        await page.pdf(
            path=str(pdf_path),
            format='A4',
            print_background=True,
            scale=0.9,  # Reduce scale for smaller file
            margin={
                'top': '0px',
                'right': '0px',
                'bottom': '0px',
                'left': '0px'
            }
        )
        
        await browser.close()
    
    initial_size = pdf_path.stat().st_size
    print(f"   📊 Initial size: {initial_size / 1024:.1f} KB")
    
    # Compress PDF using pypdf
    try:
        from pypdf import PdfReader, PdfWriter
        
        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()
        
        for page in reader.pages:
            page.compress_content_streams()
            writer.add_page(page)
        
        # Write compressed version
        compressed_path = script_dir / "temp_compressed.pdf"
        with open(str(compressed_path), 'wb') as f:
            writer.write(f)
        
        compressed_size = compressed_path.stat().st_size
        
        # Use compressed if smaller
        if compressed_size < initial_size:
            import shutil
            shutil.move(str(compressed_path), str(pdf_path))
            print(f"   🗜️ Compressed with pypdf: {compressed_size / 1024:.1f} KB")
        else:
            compressed_path.unlink()
            
    except ImportError:
        print("   ⚠️ pypdf not installed, skipping compression")
    except Exception as e:
        print(f"   ⚠️ Compression failed: {e}")
    
    final_size = pdf_path.stat().st_size
    print(f"✅ PDF successfully generated!")
    print(f"   📁 Location: {pdf_path}")
    print(f"   📊 Final size: {final_size / 1024:.1f} KB ({final_size / (1024*1024):.2f} MB)")
    
    if final_size > 2 * 1024 * 1024:
        print(f"   ⚠️ Warning: File exceeds 2MB limit!")
    else:
        print(f"   ✅ File is under 2MB limit!")
    
    return True

if __name__ == "__main__":
    asyncio.run(generate_pdf())
