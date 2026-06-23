# Step1: Import deps
from langchain_core.tools import tool
from datetime import datetime
from pathlib import Path
import subprocess
import shutil
from fpdf import FPDF

@tool
def render_latex_pdf(latex_content: str) -> str:
    """Render a LaTeX document to PDF.
    Falls back to FPDF if Tectonic is not installed.

    Args:
        latex_content: The LaTeX document content as a string
    Returns:
        Path to the generated PDF document
    """
    # Step2: Create directory
    output_dir = Path("output").absolute()
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"paper_{timestamp}.pdf"
    final_pdf = output_dir / pdf_filename

    # Step3: Check for Tectonic
    if shutil.which("tectonic"):
        try:
            tex_filename = f"paper_{timestamp}.tex"
            tex_file = output_dir / tex_filename
            tex_file.write_text(latex_content, encoding='utf-8')

            result = subprocess.run(
                ["tectonic", str(tex_file), "--outdir", str(output_dir)],
                capture_output=True,
                text=True,
            )
            
            if final_pdf.exists():
                print(f"Successfully generated LaTeX PDF at {final_pdf}")
                return str(final_pdf)
        except Exception as e:
            print(f"LaTeX rendering failed, falling back to FPDF: {e}")

    # Step4: Fallback to FPDF (Safe Mode)
    try:
        print("Using FPDF fallback...")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=11)
        
        # Clean latex commands for simple PDF view
        clean_text = latex_content.replace('\\section{', '\n\n').replace('}', '').replace('\\', '')
        
        pdf.multi_cell(0, 10, txt=clean_text.encode('latin-1', 'ignore').decode('latin-1'))
        pdf.output(str(final_pdf))
        
        return str(final_pdf)
    except Exception as e:
        return f"Error generating PDF: {str(e)}"