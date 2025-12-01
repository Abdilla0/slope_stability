"""
Gemini AI Analyzer for Slope Stability Results
Provides intelligent analysis, cost estimation, and recommendations
"""

import google.generativeai as genai
import os
from typing import Dict, List

class GeminiAnalyzer:
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.environ.get('GEMINI_API_KEY')

        if not api_key:
            raise ValueError("Gemini API key is required.")

        genai.configure(api_key=api_key)

        # FREE TIER + Supported in your list
        self.model = genai.GenerativeModel("models/gemini-flash-latest")


    def analyze_results(self, results: Dict, layers: List[Dict]) -> Dict:
        """
        Generate comprehensive AI analysis of slope stability results
        
        Args:
            results: Dictionary containing calculation results
            layers: List of lithology layers with properties
            
        Returns:
            Dictionary with AI-generated analysis sections
        """
        
        # Prepare context for AI
        prompt = self._create_analysis_prompt(results, layers)
        
        try:
            # Generate AI analysis
            response = self.model.generate_content(prompt)
            
            # Parse the response
            analysis = self._parse_ai_response(response.text)
            
            return {
                'success': True,
                'analysis': analysis,
                'raw_response': response.text
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'analysis': None
            }
    
    def _create_analysis_prompt(self, results: Dict, layers: List[Dict]) -> str:
        """Create detailed prompt for Gemini AI"""
        
        # Extract key information
        height = results.get('optimal_height', 0)
        angle = results.get('optimal_angle', 0)
        sf = results.get('safety_factor', 0)
        target_sf = results.get('target_sf', 1.3)
        water_depth = results.get('water_table_depth', 0)
        
        # Build lithology summary
        layer_summary = "\n".join([
            f"- {layer.get('name', 'Unknown')}: Depth {layer.get('depth_from', 0)}-{layer.get('depth_to', 0)}m, "
            f"Cohesion {layer.get('cohesion', 0)} kPa, Friction Angle {layer.get('friction_angle', 0)}°"
            for layer in layers
        ])
        
        prompt = f"""You are an expert mining geotechnical engineer analyzing slope stability results for an open-pit mine.

ANALYSIS CONTEXT:
==================
Optimal Design Found:
- Slope Height: {height} meters
- Slope Angle: {angle} degrees
- Safety Factor: {sf:.3f}
- Target Safety Factor: {target_sf}
- KEPMEN ESDM Compliance: {'COMPLIANT' if sf >= target_sf else 'NON-COMPLIANT'}

Site Conditions:
- Water Table Depth: {water_depth} meters from surface

Geological Stratigraphy:
{layer_summary}

ANALYSIS REQUIREMENTS:
======================
Please provide a comprehensive analysis with the following sections:

1. EXECUTIVE SUMMARY (2-3 sentences)
   - Brief overview of the optimal design
   - Key finding and compliance status

2. COST ESTIMATION (detailed breakdown)
   - Estimated excavation volume (m³)
   - Cost per cubic meter (use Indonesian mining standards)
   - Total excavation cost estimate (in USD and IDR)
   - Operating cost per meter of depth
   - Include cost comparison between this design vs alternative angles

3. REAL-WORLD COMPARISON
   - Reference similar open-pit mines in Indonesia or internationally
   - Compare slope angles used in similar geological conditions
   - Industry best practices for these lithologies
   - Mention specific mine examples if possible

4. SAFETY RECOMMENDATIONS
   - Monitoring requirements (instruments needed)
   - Inspection frequency recommendations
   - Weather-related considerations (especially rainy season)
   - Early warning system suggestions
   - Risk mitigation strategies

5. ENGINEERING INSIGHTS
   - Why this angle/height combination is optimal
   - Trade-offs analysis (safety vs economics)
   - Sensitivity to water table changes
   - Long-term stability considerations

6. IMPLEMENTATION PLAN
   - Phased excavation recommendations
   - Bench design suggestions
   - Drainage system requirements
   - Construction sequence

Format your response clearly with section headers. Use specific numbers and be practical.
Focus on Indonesian mining context where relevant (KEPMEN ESDM, local costs, typical practices).
"""
        
        return prompt
    
    def _parse_ai_response(self, response_text: str) -> Dict:
        """
        Parse AI response into structured sections
        
        Returns:
            Dictionary with analysis sections
        """
        
        sections = {
            'executive_summary': '',
            'cost_estimation': '',
            'real_world_comparison': '',
            'safety_recommendations': '',
            'engineering_insights': '',
            'implementation_plan': '',
            'full_text': response_text
        }
        
        # Try to extract sections based on headers
        current_section = None
        lines = response_text.split('\n')
        
        section_keywords = {
            'EXECUTIVE SUMMARY': 'executive_summary',
            'COST ESTIMATION': 'cost_estimation',
            'REAL-WORLD COMPARISON': 'real_world_comparison',
            'SAFETY RECOMMENDATIONS': 'safety_recommendations',
            'ENGINEERING INSIGHTS': 'engineering_insights',
            'IMPLEMENTATION PLAN': 'implementation_plan'
        }
        
        for line in lines:
            # Check if line is a section header
            line_upper = line.strip().upper()
            matched = False
            
            for keyword, section_key in section_keywords.items():
                if keyword in line_upper:
                    current_section = section_key
                    matched = True
                    break
            
            # Add content to current section
            if not matched and current_section:
                sections[current_section] += line + '\n'
        
        # Clean up sections
        for key in sections:
            if key != 'full_text':
                sections[key] = sections[key].strip()
        
        return sections
    
    def generate_quick_summary(self, results: Dict) -> str:
        """
        Generate a quick one-paragraph summary for display
        
        Args:
            results: Calculation results dictionary
            
        Returns:
            Quick summary text
        """
        
        height = results.get('optimal_height', 0)
        angle = results.get('optimal_angle', 0)
        sf = results.get('safety_factor', 0)
        
        prompt = f"""Provide a brief 2-3 sentence expert summary of this slope design:
- Height: {height}m, Angle: {angle}°, Safety Factor: {sf:.2f}

Focus on: Is this a good design? Key considerations? One practical recommendation.
Keep it concise and actionable."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except:
            return f"Optimal design: {angle}° slope at {height}m depth with SF={sf:.2f}. Meets safety standards."


# Example usage
if __name__ == "__main__":
    # Test the analyzer
    analyzer = GeminiAnalyzer(api_key="YOUR_API_KEY_HERE")
    
    test_results = {
        'optimal_height': 35.0,
        'optimal_angle': 40.0,
        'safety_factor': 1.35,
        'target_sf': 1.3,
        'water_table_depth': 8.0
    }
    
    test_layers = [
        {'name': 'Soil', 'depth_from': 0, 'depth_to': 5, 'cohesion': 90, 'friction_angle': 26.71},
        {'name': 'Mudstone', 'depth_from': 5, 'depth_to': 15, 'cohesion': 120.72, 'friction_angle': 19.95},
    ]
    
    analysis = analyzer.analyze_results(test_results, test_layers)
    
    if analysis['success']:
        print("AI Analysis Generated Successfully!")
        print("\nExecutive Summary:")
        print(analysis['analysis']['executive_summary'])
    else:
        print(f"Error: {analysis['error']}")