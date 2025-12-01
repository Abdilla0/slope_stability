# views.py
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.views import View
from django.core.files.storage import FileSystemStorage
import pandas as pd
import numpy as np
from io import BytesIO
import json
from datetime import datetime
import os

# Import our Bishop calculator and Gemini analyzer
from .bishop_calculator import BishopCalculator, SoilLayer
from .gemini_analyzer import GeminiAnalyzer

class UploadDataView(View):
    """Handle Excel file upload and initial data display"""
    
    def get(self, request):
        return render(request, 'slope/upload.html')
    
    def post(self, request):
        try:
            excel_file = request.FILES.get('excel_file')
            
            if not excel_file:
                return JsonResponse({'error': 'No file uploaded'}, status=400)
            
            # Read Excel file
            df_params = pd.read_excel(excel_file, sheet_name='Parameters')
            df_strat = pd.read_excel(excel_file, sheet_name='Stratigraphy')
            df_config = pd.read_excel(excel_file, sheet_name='Configuration')
            
            # Store data in session
            request.session['geotechnical_params'] = df_params.to_json()
            request.session['stratigraphy'] = df_strat.to_json()
            request.session['configuration'] = df_config.to_json()
            
            # Skip preview, go directly to calculation
            return redirect('calculate')
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


class ConfigureAnalysisView(View):
    """Configure analysis parameters"""
    
    def get(self, request):
        # Get data from session
        config_json = request.session.get('configuration')
        
        if config_json:
            df_config = pd.read_json(config_json)
            config = df_config.to_dict('records')[0] if len(df_config) > 0 else {}
        else:
            config = {}
        
        context = {
            'water_table_depth': config.get('water_table_depth', 10.0),
            'target_sf': config.get('target_sf', 1.3),
            'initial_angle': config.get('initial_angle', 45.0),
            'max_height': config.get('max_height', 100.0),
            'grid_points': config.get('grid_points', 10),
        }
        
        return render(request, 'slope/configure.html', context)
    
    def post(self, request):
        # Update configuration
        config = {
            'water_table_depth': float(request.POST.get('water_table_depth', 10)),
            'target_sf': float(request.POST.get('target_sf', 1.3)),
            'initial_angle': float(request.POST.get('initial_angle', 45)),
            'max_height': float(request.POST.get('max_height', 100)),
            'grid_points': int(request.POST.get('grid_points', 10)),
        }
        
        # Store in session
        df_config = pd.DataFrame([config])
        request.session['configuration'] = df_config.to_json()
        
        # Redirect to calculation
        return redirect('calculate')


class CalculateView(View):
    """Perform Bishop method calculations"""
    
    def get(self, request):
        try:
            # Retrieve data from session
            params_json = request.session.get('geotechnical_params')
            strat_json = request.session.get('stratigraphy')
            config_json = request.session.get('configuration')
            
            if not all([params_json, strat_json, config_json]):
                return redirect('upload')
            
            # Load data
            df_params = pd.read_json(params_json)
            df_strat = pd.read_json(strat_json)
            df_config = pd.read_json(config_json)
            config = df_config.to_dict('records')[0]
            
            # Debug: Print column names
            print("=" * 60)
            print("DEBUG - Stratigraphy columns:", df_strat.columns.tolist())
            print("DEBUG - Parameters columns:", df_params.columns.tolist())
            print("=" * 60)
            
            # Helper function to get column value (case-insensitive)
            def get_col_value(df, row_data, possible_names):
                for name in possible_names:
                    for col in df.columns:
                        if col.lower().strip().replace(' ', '_') == name.lower().strip().replace(' ', '_'):
                            return row_data[col]
                return None
            
            # Build soil layers from stratigraphy
            layers = []
            
            for _, row in df_strat.iterrows():
                try:
                    # Get lithology name - handle different possible column names
                    lithology_name = None
                    for col_name in df_strat.columns:
                        if col_name.lower().strip() in ['lithology', 'litology']:
                            lithology_name = row[col_name]
                            break
                    
                    if not lithology_name:
                        print(f"Warning: Could not find Lithology column in row")
                        continue
                    
                    print(f"Processing lithology: {lithology_name}")
                    
                    # Find matching parameters - case insensitive match
                    param_row = None
                    for idx, prow in df_params.iterrows():
                        param_lith = None
                        for col_name in df_params.columns:
                            if col_name.lower().strip() in ['lithology', 'litology']:
                                param_lith = prow[col_name]
                                break
                        
                        if param_lith and str(param_lith).strip().lower() == str(lithology_name).strip().lower():
                            param_row = prow
                            break
                    
                    if param_row is None:
                        print(f"ERROR: Could not find parameters for lithology: {lithology_name}")
                        continue
                    
                    # Get depth values
                    depth_from = get_col_value(df_strat, row, ['Depth_From', 'depth_from', 'DepthFrom'])
                    depth_to = get_col_value(df_strat, row, ['Depth_To', 'depth_to', 'DepthTo'])
                    
                    # Get parameter values
                    cohesion = get_col_value(df_params, param_row, ['Cohesion', 'cohesion'])
                    friction_angle = get_col_value(df_params, param_row, ['Friction_Angle', 'friction_angle', 'FrictionAngle'])
                    unit_weight_sat = get_col_value(df_params, param_row, ['Unit_Weight_Saturated', 'unit_weight_saturated', 'UnitWeightSaturated'])
                    unit_weight_unsat = get_col_value(df_params, param_row, ['Unit_Weight_Unsaturated', 'unit_weight_unsaturated', 'UnitWeightUnsaturated'])
                    
                    print(f"Found values - Depth: {depth_from}-{depth_to}m, Cohesion: {cohesion}, Friction: {friction_angle}")
                    
                    layer = SoilLayer(
                        name=lithology_name,
                        depth_from=float(depth_from),
                        depth_to=float(depth_to),
                        cohesion=float(cohesion),
                        friction_angle=float(friction_angle),
                        unit_weight_sat=float(unit_weight_sat),
                        unit_weight_unsat=float(unit_weight_unsat)
                    )
                    layers.append(layer)
                    
                except Exception as e:
                    print(f"ERROR processing layer: {e}")
                    continue
            
            print(f"Successfully created {len(layers)} layers")
            print("=" * 60)
            
            if not layers:
                return render(request, 'slope/results.html', {
                    'error': 'Could not process any layers. Please check your Excel file format and column names.'
                })
            
            # Create calculator
            calculator = BishopCalculator(
                layers=layers,
                water_table_depth=config['water_table_depth']
            )
            
            # Optimize slope design - test multiple angles to find best combination
            print("Starting slope optimization...")
            
            # Check if we should optimize both or just height
            optimize_both = config.get('optimize_both', True)
            angle_min = config.get('angle_min', 30.0)
            angle_max = config.get('angle_max', 60.0)
            
            if optimize_both:
                print(f"Optimizing BOTH angle and height (testing {angle_min}° to {angle_max}°)")
                result = calculator.optimize_slope_design(
                    target_sf=config['target_sf'],
                    initial_angle=config.get('initial_angle', 45.0),
                    max_height=config['max_height'],
                    height_step=5.0,
                    optimize_both=True,
                    angle_range=(angle_min, angle_max),
                    angle_step=5.0
                )
            else:
                print(f"Optimizing height only for fixed angle {config['initial_angle']}°")
                result = calculator.optimize_slope_design(
                    target_sf=config['target_sf'],
                    initial_angle=config['initial_angle'],
                    max_height=config['max_height'],
                    height_step=5.0,
                    optimize_both=False
                )
            
            if not result:
                return render(request, 'slope/results.html', {
                    'error': 'No safe design found with the given parameters. Try adjusting target SF or slope angle.'
                })
            
            print(f"Optimization complete! Height: {result['height']}m, Angle: {result['angle']}°, SF: {result['sf']}")
            
            # Generate AI Analysis using Gemini
            ai_analysis = None
            ai_error = None
            
            try:
                # Get Gemini API key from environment or settings
                gemini_api_key = os.environ.get('GEMINI_API_KEY')
                
                if gemini_api_key:
                    print("Generating AI analysis with Gemini...")
                    analyzer = GeminiAnalyzer(api_key=gemini_api_key)
                    
                    # Prepare layer data for AI
                    layer_data = []
                    for layer in layers:
                        layer_data.append({
                            'name': layer.name,
                            'depth_from': layer.depth_from,
                            'depth_to': layer.depth_to,
                            'cohesion': layer.cohesion,
                            'friction_angle': layer.friction_angle
                        })
                    
                    # Generate analysis
                    ai_result = analyzer.analyze_results({
                        'optimal_height': result['height'],
                        'optimal_angle': result['angle'],
                        'safety_factor': result['sf'],
                        'target_sf': config['target_sf'],
                        'water_table_depth': config['water_table_depth']
                    }, layer_data)
                    
                    if ai_result['success']:
                        ai_analysis = ai_result['analysis']
                        print("AI analysis generated successfully!")
                    else:
                        ai_error = ai_result.get('error', 'Unknown error')
                        print(f"AI analysis failed: {ai_error}")
                else:
                    ai_error = "Gemini API key not configured"
                    print("Gemini API key not found. Skipping AI analysis.")
                    
            except Exception as e:
                ai_error = str(e)
                print(f"Error generating AI analysis: {e}")
            
            # Prepare detailed slice data
            slices_data = []
            if result.get('critical_circle') and result['critical_circle'].get('slices'):
                for i, s in enumerate(result['critical_circle']['slices'], 1):
                    slices_data.append({
                        'slice_no': i,
                        'x_position': round(s.x, 2),
                        'width': round(s.width, 2),
                        'height': round(s.height, 2),
                        'weight': round(s.weight, 2),
                        'cohesion': round(s.cohesion, 2),
                        'friction_angle': round(s.friction_angle, 2),
                        'base_angle': round(s.base_angle, 2),
                    })
            
            # Store results in session for export
            request.session['results'] = {
                'optimal_height': result['height'],
                'optimal_angle': result['angle'],
                'safety_factor': result['sf'],
                'circle_center_x': result['critical_circle']['center_x'],
                'circle_center_y': result['critical_circle']['center_y'],
                'circle_radius': result['critical_circle']['radius'],
                'slices': slices_data,
                'target_sf': config['target_sf'],
                'water_table_depth': config['water_table_depth'],
                'calculation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            context = {
                'optimal_height': round(result['height'], 2),
                'optimal_angle': round(result['angle'], 2),
                'safety_factor': round(result['sf'], 3),
                'target_sf': config['target_sf'],
                'circle_center_x': round(result['critical_circle']['center_x'], 2),
                'circle_center_y': round(result['critical_circle']['center_y'], 2),
                'circle_radius': round(result['critical_circle']['radius'], 2),
                'slices_data': slices_data,
                'num_slices': len(slices_data),
                'kepmen_compliance': 'COMPLIANT' if result['sf'] >= config['target_sf'] else 'NON-COMPLIANT',
                'ai_analysis': ai_analysis,
                'ai_error': ai_error,
                'has_ai_analysis': ai_analysis is not None
            }
            
            return render(request, 'slope/results.html', context)
            
        except Exception as e:
            import traceback
            print("=" * 60)
            print("FULL ERROR TRACEBACK:")
            print(traceback.format_exc())
            print("=" * 60)
            return render(request, 'slope/results.html', {
                'error': f'Calculation error: {str(e)}'
            })


class ExportResultsView(View):
    """Export results to Excel"""
    
    def get(self, request):
        try:
            results = request.session.get('results')
            
            if not results:
                return HttpResponse('No results to export', status=400)
            
            # Create Excel file
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    'Parameter': [
                        'Optimal Slope Height (m)',
                        'Optimal Slope Angle (degrees)',
                        'Safety Factor',
                        'Target Safety Factor',
                        'KEPMEN ESDM Compliance',
                        'Water Table Depth (m)',
                        'Circle Center X (m)',
                        'Circle Center Y (m)',
                        'Circle Radius (m)',
                        'Calculation Date'
                    ],
                    'Value': [
                        results['optimal_height'],
                        results['optimal_angle'],
                        results['safety_factor'],
                        results['target_sf'],
                        'COMPLIANT' if results['safety_factor'] >= results['target_sf'] else 'NON-COMPLIANT',
                        results['water_table_depth'],
                        results['circle_center_x'],
                        results['circle_center_y'],
                        results['circle_radius'],
                        results['calculation_date']
                    ]
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Slice details sheet
                if results.get('slices'):
                    df_slices = pd.DataFrame(results['slices'])
                    df_slices.to_excel(writer, sheet_name='Slice_Details', index=False)
                
                # KEPMEN ESDM Reference sheet
                kepmen_data = {
                    'Condition': ['Overall Slope (Static)', 'Single Bench', 'Dynamic/Seismic'],
                    'Minimum SF': [1.3, 1.2, 1.1],
                    'Reference': ['KEPMEN ESDM 1827/2018', 'KEPMEN ESDM 1827/2018', 'KEPMEN ESDM 1827/2018']
                }
                df_kepmen = pd.DataFrame(kepmen_data)
                df_kepmen.to_excel(writer, sheet_name='KEPMEN_Reference', index=False)
            
            output.seek(0)
            
            # Prepare response
            response = HttpResponse(
                output.read(),
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            response['Content-Disposition'] = f'attachment; filename=slope_stability_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            
            return response
            
        except Exception as e:
            return HttpResponse(f'Export error: {str(e)}', status=500)


class DownloadTemplateView(View):
    """Download Excel template for data input"""
    
    def get(self, request):
        try:
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Parameters template
                params_template = pd.DataFrame({
                    'Lithology': ['Soil', 'Mudstone', 'Sandstone', 'Siltstone', 'Claystone'],
                    'Cohesion': [90.0, 120.72, 116.29, 151.87, 355.70],
                    'Friction_Angle': [26.71, 19.95, 30.83, 21.61, 13.82],
                    'Density_Natural': [2.25, 2.22, 2.28, 2.23, 2.16],
                    'Density_Dry': [2.11, 2.10, 2.20, 2.15, 2.11],
                    'Density_Saturated': [2.33, 2.29, 2.34, 2.30, 2.23],
                    'Unit_Weight_Natural': [22.07, 21.82, 22.37, 21.89, 21.17],
                    'Unit_Weight_Unsaturated': [20.70, 20.61, 21.59, 21.09, 20.67],
                    'Unit_Weight_Saturated': [22.86, 22.48, 22.91, 22.55, 21.84]
                })
                params_template.to_excel(writer, sheet_name='Parameters', index=False)
                
                # Stratigraphy template
                strat_template = pd.DataFrame({
                    'Layer_No': [1, 2, 3, 4],
                    'Depth_From': [0, 5, 15, 30],
                    'Depth_To': [5, 15, 30, 50],
                    'Lithology': ['Soil', 'Mudstone', 'Sandstone', 'Siltstone']
                })
                strat_template.to_excel(writer, sheet_name='Stratigraphy', index=False)
                
                # Configuration template
                config_template = pd.DataFrame({
                    'water_table_depth': [10.0],
                    'target_sf': [1.3],
                    'optimize_both': [True],
                    'angle_min': [30.0],
                    'angle_max': [60.0],
                    'initial_angle': [45.0],
                    'max_height': [100.0],
                    'grid_points': [10]
                })
                config_template.to_excel(writer, sheet_name='Configuration', index=False)
            
            output.seek(0)
            
            response = HttpResponse(
                output.read(),
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            response['Content-Disposition'] = 'attachment; filename=slope_stability_template.xlsx'
            
            return response
            
        except Exception as e:
            return HttpResponse(f'Template generation error: {str(e)}', status=500)