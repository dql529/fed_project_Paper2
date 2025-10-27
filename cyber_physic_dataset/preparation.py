import pandas as pd

# Load your dataset
dataset_path = "data_version_2.csv"  # Update this path to your actual dataset location

data = pd.read_csv(dataset_path)

data = data[data["useful"] != False]

data["temperature1"] = pd.to_numeric(data["temperature1"], errors="coerce")
data["battery1"] = pd.to_numeric(data["battery1"], errors="coerce")
data["mp_distance_x1"] = pd.to_numeric(data["mp_distance_x1"], errors="coerce")
data["temperature2"] = pd.to_numeric(data["temperature2"], errors="coerce")
data["battery2"] = pd.to_numeric(data["battery2"], errors="coerce")
data["mp_distance_x2"] = pd.to_numeric(data["mp_distance_x2"], errors="coerce")
data["temperature3"] = pd.to_numeric(data["temperature3"], errors="coerce")
data["battery3"] = pd.to_numeric(data["battery3"], errors="coerce")
data["mp_distance_x3"] = pd.to_numeric(data["mp_distance_x3"], errors="coerce")

data["temperature_change"] = abs(data["temperature3"] - data["temperature1"])
data["battery_change"] = abs(data["battery3"] - data["battery1"])
data["mp_distance_change"] = abs(data["mp_distance_x3"] - data["mp_distance_x1"])

max_temp_change = data["temperature_change"].max()
max_battery_change = data["battery_change"].max()
max_distance_change = data["mp_distance_change"].max()

data["normalized_temperature_change"] = data["temperature_change"] / max_temp_change
data["normalized_battery_change"] = data["battery_change"] / max_battery_change
data["normalized_distance_change"] = data["mp_distance_change"] / max_distance_change

# # 检查新列是否添加成功
print(
    data[
        [
            "normalized_temperature_change",
            "normalized_battery_change",
            "normalized_distance_change",
        ]
    ].head()
)

data["battery_difference"] = data["battery3"] - data["battery1"]

summary_stats = data["battery_difference"].describe()
# Get the count of unique values in battery_difference
unique_values_count = data["battery_difference"].nunique()

# Print the results
print("Summary Statistics:\n", summary_stats)
print("\nNumber of Unique Values:", unique_values_count)


# Get all unique values in battery_difference
unique_values = data["battery_difference"].unique()

# Print the unique values
print(unique_values)

columns_to_keep = [
    "wlan_fc_type",
    "wlan_duration",
    "ip_hdr_len",
    "ip_dst",
    "frame_len",
    "ip_proto",
    "frame_protocols",
    "wlan_seq",
    "data_len",
    "time_since_last_packet",
    "ip_src",
    "ip_ttl",
    "ip_id",
    "wlan_ra",
    "llc_type",
    "udp_dstport",
    "normalized_temperature_change",
    "normalized_battery_change",
    "normalized_distance_change",
    "class",
]
# Select the specified columns from the dataset
filtered_data = data[columns_to_keep]

# Optionally, save the filtered data to a new CSV file
filtered_data.to_csv("filtered_dataset.csv", index=False)

print("Filtered dataset has been saved to 'filtered_dataset.csv'.")
