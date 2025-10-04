"""
PDF Report Generator for Portfolio Optimization

Generates professional PDF reports with charts, tables, and analysis.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, PageBreak, Image, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import setup_logger

logger = setup_logger(__name__)


class PDFReportGenerator:
    """
    Generate professional PDF reports for portfolio optimization.

    Features:
    - Executive summary
    - Portfolio weights visualization
    - Performance metrics
    - Risk analysis
    - Correlation matrix
    - Historical performance
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize PDF report generator.

        Args:
            output_dir: Directory to save PDF reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))

        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12,
            spaceBefore=12
        ))

        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            alignment=TA_LEFT
        ))

    def generate_report(
        self,
        weights: pd.Series,
        metrics: Dict[str, Any],
        returns: pd.DataFrame,
        strategy: str,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate complete PDF report.

        Args:
            weights: Portfolio weights
            metrics: Performance metrics
            returns: Historical returns
            strategy: Strategy name
            filename: Output filename (optional)

        Returns:
            Path to generated PDF file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_report_{strategy.replace(' ', '_')}_{timestamp}.pdf"

        filepath = self.output_dir / filename

        logger.info(f"Generating PDF report: {filepath}")

        # Create PDF document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Build content
        story = []

        # Title page
        story.extend(self._create_title_page(strategy))
        story.append(PageBreak())

        # Executive summary
        story.extend(self._create_executive_summary(metrics, strategy))
        story.append(Spacer(1, 0.3*inch))

        # Portfolio weights
        story.extend(self._create_weights_section(weights))
        story.append(Spacer(1, 0.3*inch))

        # Performance metrics
        story.extend(self._create_metrics_section(metrics))
        story.append(PageBreak())

        # Visualizations
        story.extend(self._create_visualizations(weights, returns))

        # Build PDF
        doc.build(story)

        logger.info(f"PDF report generated successfully: {filepath}")

        return str(filepath)

    def _create_title_page(self, strategy: str) -> list:
        """Create title page elements."""
        elements = []

        # Title
        title = Paragraph(
            "Portfolio Optimization Report",
            self.styles['CustomTitle']
        )
        elements.append(Spacer(1, 2*inch))
        elements.append(title)
        elements.append(Spacer(1, 0.5*inch))

        # Strategy
        strategy_text = Paragraph(
            f"<b>Strategy:</b> {strategy}",
            self.styles['Heading2']
        )
        elements.append(strategy_text)
        elements.append(Spacer(1, 0.3*inch))

        # Date
        date_text = Paragraph(
            f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            self.styles['Normal']
        )
        elements.append(date_text)
        elements.append(Spacer(1, 1*inch))

        # Divider
        elements.append(HRFlowable(
            width="100%",
            thickness=2,
            color=colors.HexColor('#1f77b4')
        ))

        return elements

    def _create_executive_summary(self, metrics: Dict[str, Any], strategy: str) -> list:
        """Create executive summary section."""
        elements = []

        # Section header
        header = Paragraph("Executive Summary", self.styles['SectionHeader'])
        elements.append(header)

        # Summary text
        summary_text = f"""
        This report presents the results of portfolio optimization using the <b>{strategy}</b> strategy.
        The optimized portfolio consists of <b>{metrics['n_assets']} assets</b> with an expected annual
        return of <b>{metrics['return']:.2%}</b> and volatility of <b>{metrics['volatility']:.2%}</b>.
        The portfolio achieves a Sharpe ratio of <b>{metrics['sharpe']:.3f}</b>.
        """

        summary = Paragraph(summary_text, self.styles['Normal'])
        elements.append(summary)

        return elements

    def _create_weights_section(self, weights: pd.Series) -> list:
        """Create portfolio weights table."""
        elements = []

        # Section header
        header = Paragraph("Portfolio Allocation", self.styles['SectionHeader'])
        elements.append(header)

        # Prepare data
        active_weights = weights[weights > 1e-6].sort_values(ascending=False)

        table_data = [['Asset', 'Weight']]
        for asset, weight in active_weights.items():
            table_data.append([asset, f"{weight:.2%}"])

        # Add total row
        table_data.append(['<b>Total</b>', '<b>100.00%</b>'])

        # Create table
        table = Table(table_data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e0e0e0')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ]))

        elements.append(table)

        return elements

    def _create_metrics_section(self, metrics: Dict[str, Any]) -> list:
        """Create performance metrics table."""
        elements = []

        # Section header
        header = Paragraph("Performance Metrics", self.styles['SectionHeader'])
        elements.append(header)

        # Prepare metrics data
        table_data = [
            ['Metric', 'Value'],
            ['Expected Annual Return', f"{metrics['return']:.2%}"],
            ['Annual Volatility', f"{metrics['volatility']:.2%}"],
            ['Sharpe Ratio', f"{metrics['sharpe']:.3f}"],
            ['Number of Assets', str(metrics['n_assets'])]
        ]

        # Create table
        table = Table(table_data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        elements.append(table)

        return elements

    def _create_visualizations(self, weights: pd.Series, returns: pd.DataFrame) -> list:
        """Create visualization charts."""
        elements = []

        # Section header
        header = Paragraph("Portfolio Analysis", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 0.2*inch))

        # 1. Weights pie chart
        weights_chart = self._create_weights_chart(weights)
        elements.append(weights_chart)
        elements.append(Spacer(1, 0.3*inch))

        # 2. Correlation heatmap
        corr_chart = self._create_correlation_chart(returns, weights)
        elements.append(corr_chart)

        return elements

    def _create_weights_chart(self, weights: pd.Series) -> Image:
        """Create weights pie chart."""
        active_weights = weights[weights > 1e-6].sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        colors_map = plt.cm.Set3(range(len(active_weights)))

        wedges, texts, autotexts = ax.pie(
            active_weights.values,
            labels=active_weights.index,
            autopct='%1.1f%%',
            colors=colors_map,
            startangle=90
        )

        ax.set_title('Portfolio Weights Distribution', fontsize=14, fontweight='bold')

        # Save to buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()

        buf.seek(0)
        img = Image(buf, width=4.5*inch, height=3*inch)

        return img

    def _create_correlation_chart(self, returns: pd.DataFrame, weights: pd.Series) -> Image:
        """Create correlation heatmap for assets in portfolio."""
        active_assets = weights[weights > 1e-6].index
        subset_returns = returns[active_assets]

        corr_matrix = subset_returns.corr()

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
            cbar_kws={'label': 'Correlation'}
        )

        ax.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold')

        # Save to buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()

        buf.seek(0)
        img = Image(buf, width=5*inch, height=4*inch)

        return img


# Example usage
if __name__ == '__main__':
    # Sample data
    weights = pd.Series({
        'AAPL': 0.30,
        'GOOGL': 0.25,
        'MSFT': 0.20,
        'AMZN': 0.15,
        'NVDA': 0.10
    })

    metrics = {
        'return': 0.185,
        'volatility': 0.215,
        'sharpe': 0.86,
        'n_assets': 5
    }

    returns = pd.DataFrame(
        np.random.randn(252, 5) * 0.02,
        columns=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA']
    )

    # Generate report
    generator = PDFReportGenerator()
    filepath = generator.generate_report(
        weights=weights,
        metrics=metrics,
        returns=returns,
        strategy="Max Sharpe"
    )

    print(f"Report generated: {filepath}")
