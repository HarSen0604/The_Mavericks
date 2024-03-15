import React, { useState } from "react";
import jsPDF from "jspdf";

function PDFGenerator({ jsonData }) {
  const [pdfDataUrl, setPdfDataUrl] = useState(null);

  const generatePDF = () => {
    const pdf = new jsPDF();
    let yOffset = 10; // Initial y-offset for content

   
      // Format JSON data for the PDF
      const formattedData = `Item ${1}:
      tex :  hello navnee;
Acc_No: ${jsonData.Acc_No}
Amnt: ${jsonData.Amnt}
ChequeNo: ${jsonData.ChequeNo}
Sign: ${jsonData.Sign}
To: ${jsonData.To}
cheque_Img: ${jsonData.cheque_Img}
reason: ${jsonData.reason}
Status: ${jsonData.status ? "true" : "false"}
`;

      // Add formatted data to PDF
      pdf.text(20, yOffset, formattedData);
    
      // Increment y-offset for next item
      yOffset += 10; // Increase or decrease based on spacing between items
  

    // Save PDF
    const pdfFile = pdf.output("blob");
    const pdfUrl = URL.createObjectURL(pdfFile);
    setPdfDataUrl(pdfUrl);
  };

  const handleDownloadPDF = () => {
    if (pdfDataUrl) {
      const a = document.createElement("a");
      a.href = pdfDataUrl;
      a.download = "data.pdf";
      a.click();
    }
  };

  return (
    <div>
      <div>
        <h1>PDF Content</h1>
        <button onClick={generatePDF}>Generate PDF</button>
      </div>
      {pdfDataUrl && (
        <div>
          <button onClick={handleDownloadPDF}>Download PDF</button>
        </div>
      )}
    </div>
  );
}

export default PDFGenerator;
