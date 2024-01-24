import re
import pandas as pd

def parse_log_file(file_path):
    # Regular expression pattern to match the log line
    log_pattern = re.compile(r'(\d+)/5524 \[.*?\] - ETA: .*? - (.*)')

    # Function to parse a single line
    def parse_line(line):
        match = log_pattern.match(line.strip())  # strip to remove leading spaces
        if match:
            step = int(match.group(1))
            metrics = match.group(2).split(' - ')
            metrics_dict = {'step': step}
            for metric in metrics:
                try:
                    key, value = metric.split(':', 1)  # Split only on the first ':' encountered
                    metrics_dict[key.strip()] = float(value)
                except ValueError as e:
                    print()
            return metrics_dict
        return None

    # Read the log file and parse lines
    parsed_data = []
    with open(file_path, 'r') as file:
        for line in file:
            parsed_line = parse_line(line)
            if parsed_line:
                parsed_data.append(parsed_line)

    # Create DataFrame
    df = pd.DataFrame(parsed_data)

    # Remove duplicates, keeping the last entry for each step
    df = df.drop_duplicates(subset='step', keep='last')

    # Sort the DataFrame by step
    df = df.sort_values(by='step')

    # Backward fill NaN values
    df = df.bfill()

    return df

# Example usage
# file_path = 'nohup.out'
# df = parse_log_file(file_path)
# print(df)
