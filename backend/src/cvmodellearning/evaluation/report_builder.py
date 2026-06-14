import pandas as pd
import json
from pathlib import Path
from typing import Any, Dict, List, Union

# Imports for ReportLab PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# project imports
from cvmodellearning.paths import metrics_csv_path, test_cm_path, report_pdf_path, test_report_json_path, plots_dir


def _prepare_table_data(df: pd.DataFrame, include_index: bool = False) -> List[List[str]]:
    """
    Converts a DataFrame into a list of lists for reportlab.
    Includes the index as the first column if include_index is True.
    """
    # Format floats for readability before converting to string
    df_styled = df.round(4).astype(str)
    
    # Start with the header row
    header = df_styled.columns.astype(str).tolist()
    if include_index:
        # Add index name to the header
        header.insert(0, df_styled.index.name if df_styled.index.name else 'Label')
        
    data_list = [header]
    
    # Add data rows
    for index, row in df_styled.iterrows():
        row_data = row.tolist()
        if include_index:
            row_data.insert(0, str(index))
        data_list.append(row_data)
        
    return data_list

def _get_table_style() -> TableStyle:
    """Defines a clean, readable style for the tables."""
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')), 
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), (colors.white, colors.lightgrey)),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ])

def create_classification_report(job_id: str) -> Path:
    """
    Generates a multi-table PDF report from the training/test artifacts for a given job.
    """
    # 1. Resolve Paths and Check Existence
    pdf_output_path = report_pdf_path(job_id)
    pdf_output_path.parent.mkdir(parents=True, exist_ok=True) 

    try:
        # Load data artifacts
        metrics_df = pd.read_csv(metrics_csv_path(job_id))
        cm_df = pd.read_csv(test_cm_path(job_id), index_col=None, header=0)
        
        # Load structured JSON report dictionary
        with open(test_report_json_path(job_id), 'r') as f:
            report_dict = json.load(f)
            
        # Process the structured report dictionary
        df_full = pd.DataFrame(report_dict).transpose()
        
        # 1. Extract Accuracy
        overall_accuracy = report_dict.get('accuracy', 0.0)
        
        # 2. Extract Class Metrics (Exclude averages and accuracy)
        class_rows = [k for k in report_dict if k not in ['accuracy', 'macro avg', 'weighted avg']]
        df_classes = df_full.loc[class_rows]
        df_classes.index.name = 'Class'
        
        # 3. Extract Average Metrics (Only macro and weighted avg)
        avg_rows = ['macro avg', 'weighted avg']
        df_averages = df_full.loc[[k for k in avg_rows if k in df_full.index]]
        df_averages.index.name = 'Summary'
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required artifact file missing for PDF generation: {e}")

    # 2. Setup PDF Document
    doc = SimpleDocTemplate(str(pdf_output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    elements: List[Any] = []
    
    # --- Title ---
    title_style = styles['Title']
    title_style.alignment = 1 # Center align
    elements.append(Paragraph(f"Classification Pipeline Report", title_style))
    elements.append(Paragraph(f"Job ID: {job_id}", styles['h3']))
    elements.append(Spacer(1, 24))

    # --- Section 1: Training Metrics History ---
    elements.append(Paragraph("1. Training & Validation Metrics History", styles['h2']))
    elements.append(Spacer(1, 12))
    metrics_data = _prepare_table_data(metrics_df)
    t_metrics = Table(metrics_data, style=_get_table_style())
    elements.append(t_metrics)
    elements.append(Spacer(1, 24))

    # --- Section 2: Test Confusion Matrix ---
    elements.append(Paragraph("2. Test Confusion Matrix", styles['h2']))
    elements.append(Spacer(1, 12))
    
    # Prepare CM data: Add index labels to the left column
    cm_labels = cm_df.columns.tolist()
    cm_data = _prepare_table_data(cm_df)
    cm_data[0].insert(0, 'True ↓ / Predicted →') # Add header for the first column
    for i, row in enumerate(cm_data[1:]):
        row.insert(0, cm_labels[i]) # Add True Label names
        
    t_cm = Table(cm_data, style=_get_table_style(), colWidths=[1.5*inch] + [0.8*inch] * (len(cm_df.columns)))
    elements.append(t_cm)
    elements.append(Spacer(1, 24))

    elements.append(PageBreak())

    # --- Section 3: Detailed Test Classification Report (Two Tables) ---
    elements.append(Paragraph("3. Detailed Test Classification Report", styles['h2']))
    elements.append(Spacer(1, 12))
    
    # 3.1 Display Overall Accuracy
    elements.append(Paragraph(f"Overall Accuracy: <b>{overall_accuracy:.4f}</b>", styles['h3']))
    elements.append(Spacer(1, 12))
    
    report_data_classes = _prepare_table_data(df_classes, include_index=True)
    t_classes = Table(report_data_classes, style=_get_table_style(), colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
    elements.append(t_classes)
    elements.append(Spacer(1, 12))
    
    report_data_averages = _prepare_table_data(df_averages, include_index=True)
    t_averages = Table(report_data_averages, style=_get_table_style(), colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
    elements.append(t_averages)
    elements.append(Spacer(1, 24))
    
    # 3. Build PDF
    doc.build(elements)
    
    return pdf_output_path


def create_detection_report(job_id: str, results: Dict[str, Union[float, str]], model_name: str) -> Path:
    """
    Generates a PDF report summarizing the object detection evaluation results.
    Includes table of metrics and embeds validation plots (confusion matrix, curves).
    """
    pdf_output_path = report_pdf_path(job_id)
    pdf_output_path.parent.mkdir(parents=True, exist_ok=True) 

    # 1. Setup PDF Document
    doc = SimpleDocTemplate(str(pdf_output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    elements: List[Any] = []
    
    # --- Title ---
    title_style = styles['Title']
    title_style.alignment = 1 # Center align
    elements.append(Paragraph(f"Object Detection Pipeline Report", title_style))
    elements.append(Paragraph(f"Job ID: {job_id}", styles['h3']))
    elements.append(Paragraph(f"Model: {model_name}", styles['h3']))
    elements.append(Spacer(1, 24))

    # --- Section 1: Final Test Evaluation Metrics ---
    elements.append(Paragraph("1. Final Test Evaluation Metrics", styles['h2']))
    elements.append(Spacer(1, 12))
    
    # Convert results dictionary to a DataFrame
    metric_data = {
        k: [v] for k, v in results.items() 
        if isinstance(v, (float, int)) and not (isinstance(v, str) and 'dir' in k)
    }
    
    df_metrics = pd.DataFrame(metric_data).T
    df_metrics.columns = ['Value']
    df_metrics.index.name = 'Metric'

    metrics_data_list = _prepare_table_data(df_metrics, include_index=True)
    
    t_metrics = Table(metrics_data_list, 
                      style=_get_table_style(), 
                      colWidths=[2.5*inch, 1.5*inch])
    elements.append(t_metrics)
    elements.append(Spacer(1, 24))


    # --- Section 3: Training Visualizations ---
    elements.append(PageBreak()) # Start visualizations on a new page
    elements.append(Paragraph("2. Training Visualizations", styles['h2']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("The following plots were generated during training/validation. Each plot is on a separate page for clarity:", styles['Normal']))
    elements.append(Spacer(1, 12))

    # List of preferred plots to include in order
    preferred_plots = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "F1_curve.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "labels.jpg", # Sometimes useful to see data distribution
        "labels_correlogram.jpg"
    ]
    
    vis_dir = plots_dir(job_id)
    
    # We allow .png and .jpg for the files in the list
    images_found = False
    
    if vis_dir.exists():
        # Iterate through preferred list first
        for plot_name in preferred_plots:
            # Check for exact name match (or partial match if needed)
            # We check both png and jpg just in case extensions vary by version
            candidates = list(vis_dir.glob(plot_name))
            if not candidates:
                 # Check if extension might be different in list vs reality
                 stem = Path(plot_name).stem
                 candidates = list(vis_dir.glob(f"{stem}.*"))

            if candidates:
                img_path = candidates[0]
                try:
                    # ADD PAGE BREAK BEFORE EACH NEW VISUALIZATION
                    elements.append(PageBreak())

                    # Create ReportLab Image
                    # Resize to fit within margins (Letter width ~8.5", margins ~1" each => max ~6.5")
                    img = Image(str(img_path))
                    
                    # Simple aspect ratio scaling
                    max_width = 6.5 * inch
                    max_height = 9.5 * inch # Increased max height slightly for full page

                    img_width = img.drawWidth
                    img_height = img.drawHeight
                    
                    # Apply scaling logic
                    current_width = img_width
                    current_height = img_height
                    
                    if current_width > max_width:
                        ratio = max_width / current_width
                        current_width *= ratio
                        current_height *= ratio
                    
                    if current_height > max_height:
                        ratio = max_height / current_height
                        current_width *= ratio
                        current_height *= ratio
                        
                    img.drawWidth = current_width
                    img.drawHeight = current_height

                    elements.append(Paragraph(f"<b>Visualization: {plot_name}</b>", styles['h3']))
                    elements.append(Spacer(1, 6))
                    elements.append(img)
                    elements.append(Spacer(1, 24))
                    images_found = True
                    
                except Exception as e:
                    print(f"Error adding image {plot_name} to PDF: {e}")

    if not images_found:
        # If the first '2. Training Visualizations' heading was already added, 
        # but no images were found, this message should be included.
        # Note: The heading and intro text are added before the loop starts.
        # We can remove the previously added items to replace with this message 
        # or just append the message if the loop found nothing.
        # A simpler approach is to append the message, or handle the heading 
        # and intro text inside the 'if vis_dir.exists():' block.
        
        # Given the original code's structure, we append the message if no images were found
        elements.append(Paragraph("No visualization plots found in artifacts/plots.", styles['Normal']))

    # 2. Build PDF
    doc.build(elements)
    
    return pdf_output_path