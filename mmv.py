from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_pdf_from_string():
    c = canvas.Canvas("pdf_file_1", pagesize=letter)
    textobject = c.beginText()
    textobject.setTextOrigin(50, 750)  # Adjust position as needed
    textobject.setFont("Helvetica", 12)  # Adjust font and size as needed
    for line in text.split('\n'):
        textobject.textLine(line)
    c.drawText(textobject)
    c.save()
    create_pdf_from_string("example.pdf", text)

# Example usage:

