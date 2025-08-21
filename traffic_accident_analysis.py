# traffic_accident_complete.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up the style for plots
plt.style.use('default')
sns.set_palette("viridis")

class TrafficAccidentAnalyzer:
    def __init__(self):
        self.df = None
        
    def generate_sample_data(self, num_records=1000):
        """Generate realistic sample traffic accident data"""
        print("Generating sample traffic accident data...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate dates for the past year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start_date, end_date, freq='H')
        
        # Sample data parameters
        road_types = ['Highway', 'City Street', 'Residential', 'Rural Road', 'Intersection']
        weather_conditions = ['Clear', 'Rain', 'Snow', 'Fog', 'Cloudy', 'Windy']
        road_conditions = ['Dry', 'Wet', 'Icy', 'Snowy', 'Flooded']
        light_conditions = ['Daylight', 'Dark - Lighted', 'Dark - Not Lighted', 'Dawn', 'Dusk']
        severity_levels = ['Fatal', 'Serious Injury', 'Minor Injury', 'Property Damage']
        
        # Generate sample data
        data = {
            'date_time': np.random.choice(dates, num_records),
            'latitude': np.random.uniform(37.5, 37.8, num_records),  # SF Bay Area coordinates
            'longitude': np.random.uniform(-122.5, -122.2, num_records),
            'road_type': np.random.choice(road_types, num_records, p=[0.3, 0.4, 0.15, 0.1, 0.05]),
            'weather': np.random.choice(weather_conditions, num_records, p=[0.6, 0.15, 0.05, 0.05, 0.1, 0.05]),
            'road_condition': np.random.choice(road_conditions, num_records, p=[0.7, 0.15, 0.05, 0.05, 0.05]),
            'light_condition': np.random.choice(light_conditions, num_records, p=[0.5, 0.2, 0.1, 0.1, 0.1]),
            'severity': np.random.choice(severity_levels, num_records, p=[0.05, 0.15, 0.3, 0.5]),
            'vehicles_involved': np.random.randint(1, 5, num_records),
            'injuries': np.random.randint(0, 5, num_records),
            'fatalities': np.random.choice([0, 1, 2, 3], num_records, p=[0.9, 0.07, 0.02, 0.01])
        }
        
        # Create correlations
        df = pd.DataFrame(data)
        
        # More accidents during rush hours (7-9 AM, 4-6 PM)
        df['hour'] = df['date_time'].dt.hour
        rush_hour_mask = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 16) & (df['hour'] <= 18))
        df.loc[rush_hour_mask, 'severity'] = np.random.choice(severity_levels, rush_hour_mask.sum(), 
                                                           p=[0.08, 0.2, 0.35, 0.37])
        
        # More severe accidents in bad weather
        bad_weather = df['weather'].isin(['Rain', 'Snow', 'Fog'])
        df.loc[bad_weather, 'severity'] = np.random.choice(severity_levels, bad_weather.sum(), 
                                                         p=[0.1, 0.25, 0.35, 0.3])
        
        # More accidents on weekends
        df['is_weekend'] = df['date_time'].dt.dayofweek >= 5
        weekend_mask = df['is_weekend']
        df.loc[weekend_mask, 'vehicles_involved'] = np.random.randint(1, 6, weekend_mask.sum())
        
        self.df = df
        print(f"Generated {num_records} sample accident records")
        return df
    
    def preprocess_data(self):
        """Preprocess and enrich the accident data"""
        print("Preprocessing data...")
        
        # Extract time features
        self.df['hour'] = self.df['date_time'].dt.hour
        self.df['day_of_week'] = self.df['date_time'].dt.day_name()
        self.df['month'] = self.df['date_time'].dt.month_name()
        self.df['time_of_day'] = pd.cut(self.df['hour'], 
                                       bins=[0, 6, 12, 18, 24], 
                                       labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                       include_lowest=True)
        
        # Create severity score
        severity_map = {'Property Damage': 1, 'Minor Injury': 2, 'Serious Injury': 3, 'Fatal': 4}
        self.df['severity_score'] = self.df['severity'].map(severity_map)
        
        return self.df
    
    def analyze_time_patterns(self):
        """Analyze accident patterns by time"""
        print("Analyzing time patterns...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accidents by hour
        hourly_accidents = self.df.groupby('hour').size()
        axes[0, 0].bar(hourly_accidents.index, hourly_accidents.values)
        axes[0, 0].set_title('Accidents by Hour of Day')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Number of Accidents')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Accidents by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_accidents = self.df.groupby('day_of_week').size().reindex(day_order)
        axes[0, 1].bar(range(len(daily_accidents)), daily_accidents.values)
        axes[0, 1].set_title('Accidents by Day of Week')
        axes[0, 1].set_xlabel('Day')
        axes[0, 1].set_ylabel('Number of Accidents')
        axes[0, 1].set_xticks(range(len(daily_accidents)))
        axes[0, 1].set_xticklabels(daily_accidents.index, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Accidents by month
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_accidents = self.df.groupby('month').size().reindex(month_order)
        axes[1, 0].bar(range(len(monthly_accidents)), monthly_accidents.values)
        axes[1, 0].set_title('Accidents by Month')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Number of Accidents')
        axes[1, 0].set_xticks(range(len(monthly_accidents)))
        axes[1, 0].set_xticklabels(monthly_accidents.index, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Severity by time of day
        severity_by_time = self.df.groupby('time_of_day')['severity_score'].mean()
        axes[1, 1].bar(severity_by_time.index, severity_by_time.values)
        axes[1, 1].set_title('Average Severity by Time of Day')
        axes[1, 1].set_xlabel('Time of Day')
        axes[1, 1].set_ylabel('Average Severity Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('time_patterns_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.df
    
    def analyze_weather_road_conditions(self):
        """Analyze impact of weather and road conditions"""
        print("Analyzing weather and road conditions...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accidents by weather condition
        weather_accidents = self.df.groupby('weather').size().sort_values(ascending=False)
        axes[0, 0].bar(range(len(weather_accidents)), weather_accidents.values)
        axes[0, 0].set_title('Accidents by Weather Condition')
        axes[0, 0].set_xlabel('Weather Condition')
        axes[0, 0].set_ylabel('Number of Accidents')
        axes[0, 0].set_xticks(range(len(weather_accidents)))
        axes[0, 0].set_xticklabels(weather_accidents.index, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Severity by weather condition
        severity_by_weather = self.df.groupby('weather')['severity_score'].mean().sort_values(ascending=False)
        axes[0, 1].bar(range(len(severity_by_weather)), severity_by_weather.values)
        axes[0, 1].set_title('Average Severity by Weather Condition')
        axes[0, 1].set_xlabel('Weather Condition')
        axes[0, 1].set_ylabel('Average Severity Score')
        axes[0, 1].set_xticks(range(len(severity_by_weather)))
        axes[0, 1].set_xticklabels(severity_by_weather.index, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Accidents by road condition
        road_accidents = self.df.groupby('road_condition').size().sort_values(ascending=False)
        axes[1, 0].bar(range(len(road_accidents)), road_accidents.values)
        axes[1, 0].set_title('Accidents by Road Condition')
        axes[1, 0].set_xlabel('Road Condition')
        axes[1, 0].set_ylabel('Number of Accidents')
        axes[1, 0].set_xticks(range(len(road_accidents)))
        axes[1, 0].set_xticklabels(road_accidents.index, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Severity by road condition
        severity_by_road = self.df.groupby('road_condition')['severity_score'].mean().sort_values(ascending=False)
        axes[1, 1].bar(range(len(severity_by_road)), severity_by_road.values)
        axes[1, 1].set_title('Average Severity by Road Condition')
        axes[1, 1].set_xlabel('Road Condition')
        axes[1, 1].set_ylabel('Average Severity Score')
        axes[1, 1].set_xticks(range(len(severity_by_road)))
        axes[1, 1].set_xticklabels(severity_by_road.index, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('weather_road_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.df
    
    def analyze_road_types(self):
        """Analyze accident patterns by road type"""
        print("Analyzing road types...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Accidents by road type
        road_accidents = self.df.groupby('road_type').size().sort_values(ascending=False)
        axes[0].bar(range(len(road_accidents)), road_accidents.values)
        axes[0].set_title('Accidents by Road Type')
        axes[0].set_xlabel('Road Type')
        axes[0].set_ylabel('Number of Accidents')
        axes[0].set_xticks(range(len(road_accidents)))
        axes[0].set_xticklabels(road_accidents.index, rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Severity by road type
        severity_by_road = self.df.groupby('road_type')['severity_score'].mean().sort_values(ascending=False)
        axes[1].bar(range(len(severity_by_road)), severity_by_road.values)
        axes[1].set_title('Average Severity by Road Type')
        axes[1].set_xlabel('Road Type')
        axes[1].set_ylabel('Average Severity Score')
        axes[1].set_xticks(range(len(severity_by_road)))
        axes[1].set_xticklabels(severity_by_road.index, rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('road_type_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.df
    
    def analyze_light_conditions(self):
        """Analyze impact of light conditions"""
        print("Analyzing light conditions...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Accidents by light condition
        light_accidents = self.df.groupby('light_condition').size().sort_values(ascending=False)
        axes[0].bar(range(len(light_accidents)), light_accidents.values)
        axes[0].set_title('Accidents by Light Condition')
        axes[0].set_xlabel('Light Condition')
        axes[0].set_ylabel('Number of Accidents')
        axes[0].set_xticks(range(len(light_accidents)))
        axes[0].set_xticklabels(light_accidents.index, rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Severity by light condition
        severity_by_light = self.df.groupby('light_condition')['severity_score'].mean().sort_values(ascending=False)
        axes[1].bar(range(len(severity_by_light)), severity_by_light.values)
        axes[1].set_title('Average Severity by Light Condition')
        axes[1].set_xlabel('Light Condition')
        axes[1].set_ylabel('Average Severity Score')
        axes[1].set_xticks(range(len(severity_by_light)))
        axes[1].set_xticklabels(severity_by_light.index, rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('light_condition_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.df
    
    def create_hotspot_map(self):
        """Create a hotspot visualization using matplotlib"""
        print("Creating accident hotspot visualization...")
        
        plt.figure(figsize=(12, 8))
        
        # Create a scatter plot with color based on severity
        scatter = plt.scatter(self.df['longitude'], self.df['latitude'], 
                            c=self.df['severity_score'], 
                            cmap='Reds', alpha=0.6, s=50)
        
        plt.colorbar(scatter, label='Severity Score')
        plt.title('Accident Hotspots by Location and Severity')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        
        # Add text annotation for the area
        plt.text(0.02, 0.98, 'San Francisco Bay Area', transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('accident_hotspots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.df
    
    def correlation_analysis(self):
        """Perform correlation analysis between factors"""
        print("Performing correlation analysis...")
        
        # Create encoded data for correlation
        encoded_df = self.df.copy()
        
        # Encode categorical variables
        categorical_cols = ['road_type', 'weather', 'road_condition', 'light_condition', 'severity']
        for col in categorical_cols:
            encoded_df[col] = encoded_df[col].astype('category').cat.codes
        
        # Calculate correlation matrix
        correlation_matrix = encoded_df[['road_type', 'weather', 'road_condition', 
                                       'light_condition', 'severity_score', 'hour',
                                       'vehicles_involved', 'injuries', 'fatalities']].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': .8})
        plt.title('Correlation Matrix of Accident Factors')
        plt.tight_layout()
        plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_matrix
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("Generating comprehensive report...")
        
        report = {
            'total_accidents': len(self.df),
            'fatal_accidents': len(self.df[self.df['severity'] == 'Fatal']),
            'injury_accidents': len(self.df[self.df['severity'].isin(['Serious Injury', 'Minor Injury'])]),
            'property_damage_only': len(self.df[self.df['severity'] == 'Property Damage']),
            'most_dangerous_hour': self.df.groupby('hour').size().idxmax(),
            'most_dangerous_day': self.df.groupby('day_of_week').size().idxmax(),
            'most_common_weather': self.df['weather'].value_counts().idxmax(),
            'most_common_road_condition': self.df['road_condition'].value_counts().idxmax(),
            'most_common_light_condition': self.df['light_condition'].value_counts().idxmax(),
            'highest_severity_weather': self.df.groupby('weather')['severity_score'].mean().idxmax(),
            'highest_severity_road_type': self.df.groupby('road_type')['severity_score'].mean().idxmax(),
            'highest_severity_light_condition': self.df.groupby('light_condition')['severity_score'].mean().idxmax()
        }
        
        # Print report
        print("\n" + "="*60)
        print("TRAFFIC ACCIDENT ANALYSIS REPORT")
        print("="*60)
        print(f"Total accidents analyzed: {report['total_accidents']}")
        print(f"Fatal accidents: {report['fatal_accidents']} ({report['fatal_accidents']/report['total_accidents']*100:.1f}%)")
        print(f"Injury accidents: {report['injury_accidents']} ({report['injury_accidents']/report['total_accidents']*100:.1f}%)")
        print(f"Property damage only: {report['property_damage_only']} ({report['property_damage_only']/report['total_accidents']*100:.1f}%)")
        print(f"\nMost dangerous hour: {report['most_dangerous_hour']}:00")
        print(f"Most dangerous day: {report['most_dangerous_day']}")
        print(f"Most common weather condition: {report['most_common_weather']}")
        print(f"Most common road condition: {report['most_common_road_condition']}")
        print(f"Most common light condition: {report['most_common_light_condition']}")
        print(f"Weather with highest severity: {report['highest_severity_weather']}")
        print(f"Road type with highest severity: {report['highest_severity_road_type']}")
        print(f"Light condition with highest severity: {report['highest_severity_light_condition']}")
        
        # Additional insights
        print(f"\n=== ADDITIONAL INSIGHTS ===")
        rush_hour_accidents = len(self.df[((self.df['hour'] >= 7) & (self.df['hour'] <= 9)) | 
                                       ((self.df['hour'] >= 16) & (self.df['hour'] <= 18))])
        print(f"Rush hour accidents (7-9AM, 4-6PM): {rush_hour_accidents} ({rush_hour_accidents/report['total_accidents']*100:.1f}%)")
        
        weekend_accidents = len(self.df[self.df['date_time'].dt.dayofweek >= 5])
        print(f"Weekend accidents: {weekend_accidents} ({weekend_accidents/report['total_accidents']*100:.1f}%)")
        
        bad_weather_accidents = len(self.df[self.df['weather'].isin(['Rain', 'Snow', 'Fog'])])
        print(f"Bad weather accidents: {bad_weather_accidents} ({bad_weather_accidents/report['total_accidents']*100:.1f}%)")
        
        return report

def main():
    """Main function to run the traffic accident analysis"""
    print("=== TRAFFIC ACCIDENT PATTERN ANALYSIS ===")
    print("Analyzing patterns related to road conditions, weather, and time of day")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = TrafficAccidentAnalyzer()
    
    # Generate sample data
    analyzer.generate_sample_data(1000)
    
    # Preprocess data
    analyzer.preprocess_data()
    
    # Perform analyses
    analyzer.analyze_time_patterns()
    analyzer.analyze_weather_road_conditions()
    analyzer.analyze_road_types()
    analyzer.analyze_light_conditions()
    analyzer.correlation_analysis()
    
    # Create hotspot visualization
    analyzer.create_hotspot_map()
    
    # Generate final report
    report = analyzer.generate_report()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Generated files:")
    print("- time_patterns_analysis.png (Time-based patterns)")
    print("- weather_road_analysis.png (Weather & road conditions)")
    print("- road_type_analysis.png (Road type analysis)")
    print("- light_condition_analysis.png (Light conditions)")
    print("- correlation_analysis.png (Factor correlations)")
    print("- accident_hotspots.png (Accident hotspots)")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()