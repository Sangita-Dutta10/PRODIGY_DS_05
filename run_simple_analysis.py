# run_simple_analysis.py
from traffic_accident_analysis import TrafficAccidentAnalyzer

def run_complete_analysis():
    """Run the complete traffic accident analysis"""
    print("Starting complete traffic accident analysis...")
    
    analyzer = TrafficAccidentAnalyzer()
    
    # Generate and preprocess data
    analyzer.generate_sample_data(800)
    analyzer.preprocess_data()
    
    # Run all analyses
    print("\n1. Analyzing time patterns...")
    analyzer.analyze_time_patterns()
    
    print("\n2. Analyzing weather and road conditions...")
    analyzer.analyze_weather_road_conditions()
    
    print("\n3. Analyzing road types...")
    analyzer.analyze_road_types()
    
    print("\n4. Analyzing light conditions...")
    analyzer.analyze_light_conditions()
    
    print("\n5. Performing correlation analysis...")
    analyzer.correlation_analysis()
    
    print("\n6. Creating hotspot visualization...")
    analyzer.create_hotspot_map()
    
    print("\n7. Generating final report...")
    analyzer.generate_report()
    
    print("\nAnalysis completed successfully!")
    print("Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    run_complete_analysis()