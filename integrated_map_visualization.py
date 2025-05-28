
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Load ZHVI data
zhvi_df = pd.read_csv("sd_zhvi_2000_to_2025_nhbd.csv")
zhvi_df.drop(['RegionType', 'StateName', 'State', 'City', 'Metro', 'CountyName'], axis=1, inplace=True)

# Load transit stops data
transit_df = pd.read_csv("transit_stops_datasd.csv")
transit_df = transit_df[transit_df['stop_agncy'] == 'MTS']
transit_df = transit_df.drop(['stop_uid', 'stop_lat', 'stop_lon', 'stop_agncy', 'stop_code', 'stop_id', 'wheelchair', 'intersec', 'stop_place', 'parent_sta'], axis=1)

# Create GeoDataFrame for transit stops
transit_gdf = gpd.GeoDataFrame(
    transit_df, 
    geometry=gpd.points_from_xy(x=transit_df.lng, y=transit_df.lat), 
    crs="EPSG:4326"
)

print(f"Loaded {len(transit_gdf)} transit stops")

# Load neighborhood shapefiles
try:
    neighborhoods = gpd.read_file("SDPD_Beats_shapefile/SDPD_Beats.shp")
    zones = gpd.read_file("Zoning_Base_SD_shapefile/Zoning_Base_SD.shp")
    
    # Remove restricted zones (AR-1-1, AG-1-1, AR-1-2)
    uncounted_zones = zones[zones["ZONE_NAME"].isin(["AR-1-1", "AG-1-1", "AR-1-2"])]
    
    # Convert to consistent CRS
    transit_gdf = transit_gdf.to_crs(epsg=26911)
    neighborhoods = neighborhoods.to_crs(epsg=26911)
    uncounted_zones = uncounted_zones.to_crs(epsg=26911)
    
    # Clean neighborhoods by removing restricted zones
    neighborhoods_cleaned = gpd.overlay(neighborhoods, uncounted_zones, how='difference')
    
    print(f"Loaded {len(neighborhoods_cleaned)} cleaned neighborhoods")
    
except Exception as e:
    print(f"Could not load shapefiles: {e}")
    print("Creating simplified neighborhood boundaries from existing data...")
    
    # Fallback: Create simplified boundaries if shapefiles aren't available
    # This is a simplified approach - in practice you'd use the actual shapefiles
    neighborhoods_cleaned = None

# Calculate transit density (stops within 850m of neighborhood centroids)
if neighborhoods_cleaned is not None:
    neighborhoods_cleaned["buffer_850m"] = neighborhoods_cleaned.geometry.centroid.buffer(850)
    buffered = gpd.GeoDataFrame(neighborhoods_cleaned, geometry="buffer_850m")
    joined = gpd.sjoin(transit_gdf, buffered, predicate="within", how="inner")
    stop_counts = joined.groupby("NAME").size().reset_index(name="stop_count_850m")
    neighborhoods_cleaned = neighborhoods_cleaned.merge(stop_counts, on="NAME", how="left")
    neighborhoods_cleaned["stop_count_850m"] = neighborhoods_cleaned["stop_count_850m"].fillna(0)

# Get latest ZHVI values for neighborhood coloring
latest_zhvi_col = zhvi_df.columns[-1]  # Most recent date column
zhvi_latest = zhvi_df[['RegionName', latest_zhvi_col]].copy()
zhvi_latest.columns = ['RegionName', 'ZHVI_Latest']
zhvi_latest = zhvi_latest.dropna()

print(f"Using ZHVI data from {latest_zhvi_col}")

# Create the comprehensive map
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('San Diego Housing and Transit Analysis', fontsize=20, fontweight='bold')

# Map 1: Transit Density by Neighborhood
ax1 = axes[0, 0]
if neighborhoods_cleaned is not None:
    neighborhoods_cleaned.plot(
        column="stop_count_850m", 
        cmap="viridis", 
        legend=True, 
        ax=ax1,
        edgecolor='white',
        linewidth=0.5,
        legend_kwds={'label': 'Transit Stops (850m radius)', 'shrink': 0.8}
    )
    
    # Overlay transit stops
    transit_gdf.plot(ax=ax1, color='red', markersize=8, alpha=0.7, label='Transit Stops')
    
ax1.set_title('Transit Density by Neighborhood', fontsize=14, fontweight='bold')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.legend()
ax1.axis('off')

# Map 2: Transit Stops Distribution
ax2 = axes[0, 1]
transit_gdf_geo = transit_gdf.to_crs(epsg=4326)  # Convert back to lat/lon for display
transit_gdf_geo.plot(ax=ax2, color='blue', markersize=12, alpha=0.6)

if neighborhoods_cleaned is not None:
    neighborhoods_display = neighborhoods_cleaned.to_crs(epsg=4326)
    neighborhoods_display.plot(ax=ax2, facecolor='lightgray', edgecolor='black', alpha=0.3, linewidth=0.5)

ax2.set_title('San Diego Transit Stop Locations (2025)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.grid(True, alpha=0.3)

# Map 3: ZHVI Trends for Selected Neighborhoods
ax3 = axes[1, 0]
selected_neighborhoods = ['Mira Mesa', 'Rancho Penasquitos', 'Carmel Valley', 'University City', 'Rancho Bernardo']
zhvi_subset = zhvi_df[zhvi_df['RegionName'].isin(selected_neighborhoods)]

if not zhvi_subset.empty:
    # Melt data for plotting
    date_columns = [col for col in zhvi_df.columns if col not in ['RegionID', 'SizeRank', 'RegionName']]
    zhvi_melted = zhvi_subset.melt(
        id_vars=['RegionName'], 
        value_vars=date_columns,
        var_name='Date', 
        value_name='ZHVI'
    )
    zhvi_melted['Date'] = pd.to_datetime(zhvi_melted['Date'])
    zhvi_melted = zhvi_melted.dropna()
    
    # Plot trends
    for neighborhood in selected_neighborhoods:
        data = zhvi_melted[zhvi_melted['RegionName'] == neighborhood]
        if not data.empty:
            ax3.plot(data['Date'], data['ZHVI'], marker='o', markersize=2, label=neighborhood, linewidth=2)

ax3.set_title('ZHVI Trends for Selected Neighborhoods (2000-2025)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Year')
ax3.set_ylabel('ZHVI ($)')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# Map 4: Neighborhoods with Zero Transit Access
ax4 = axes[1, 1]
if neighborhoods_cleaned is not None:
    # Highlight neighborhoods with no transit stops
    no_transit = neighborhoods_cleaned[neighborhoods_cleaned['stop_count_850m'] == 0].to_crs(epsg=4326)
    all_neighborhoods = neighborhoods_cleaned.to_crs(epsg=4326)
    
    # Plot all neighborhoods in light gray
    all_neighborhoods.plot(ax=ax4, facecolor='lightgray', edgecolor='black', alpha=0.5, linewidth=0.5)
    
    # Highlight zero-transit neighborhoods in red
    if not no_transit.empty:
        no_transit.plot(ax=ax4, facecolor='red', edgecolor='darkred', alpha=0.8, linewidth=1)
        
        # Add labels for some of the larger zero-transit areas
        for idx, row in no_transit.iterrows():
            if row['NAME'] and row['NAME'] != 'None':
                centroid = row.geometry.centroid
                ax4.annotate(row['NAME'], (centroid.x, centroid.y), 
                           fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

ax4.set_title('Neighborhoods with Limited Transit Access', fontsize=14, fontweight='bold')
ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')

plt.tight_layout()
plt.show()

# Summary Statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"\nTransit Infrastructure:")
print(f"  Total MTS Transit Stops: {len(transit_gdf)}")

if neighborhoods_cleaned is not None:
    print(f"  Total Neighborhoods Analyzed: {len(neighborhoods_cleaned)}")
    print(f"  Neighborhoods with No Transit Access: {len(neighborhoods_cleaned[neighborhoods_cleaned['stop_count_850m'] == 0])}")
    print(f"  Average Transit Stops per Neighborhood: {neighborhoods_cleaned['stop_count_850m'].mean():.1f}")
    print(f"  Max Transit Stops in Single Neighborhood: {neighborhoods_cleaned['stop_count_850m'].max():.0f}")
    
    print(f"\nNeighborhoods with Zero Transit Access:")
    zero_transit = neighborhoods_cleaned[neighborhoods_cleaned['stop_count_850m'] == 0]
    for idx, row in zero_transit.iterrows():
        if row['NAME'] and row['NAME'] != 'None':
            print(f"  - {row['NAME']}")

print(f"\nHousing Market (ZHVI):")
print(f"  Total Neighborhoods with ZHVI Data: {len(zhvi_df)}")
print(f"  Latest ZHVI Date: {latest_zhvi_col}")

if not zhvi_latest.empty:
    print(f"  Median Home Value (Latest): ${zhvi_latest['ZHVI_Latest'].median():,.0f}")
    print(f"  Highest Home Value: ${zhvi_latest['ZHVI_Latest'].max():,.0f}")
    print(f"  Lowest Home Value: ${zhvi_latest['ZHVI_Latest'].min():,.0f}")

print("\n" + "="*60)
