import csv

# Read vocabulary file and create a mapping from mid to class number
vocabulary_file = "../../fsd50k_data/raw/metadata/vocabulary.csv"
metadata_file = "../../fsd50k_data/raw/metadata/metadata.csv"
new_metadata_file = "../../fsd50k_data/raw/metadata/new_metadata_test.csv"
vocabulary_map = {}

with open(vocabulary_file, mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        class_number = rows[0]
        mid = rows[2]
        vocabulary_map[mid] = class_number

with open(metadata_file, mode='r') as infile, open(new_metadata_file, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ["split"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in reader:
        mids = row['mids'].split(',')
        class_numbers = [vocabulary_map[mid] for mid in mids]
        row['mids'] = ','.join(class_numbers)
        row['split'] = "test"
        writer.writerow(row)

print("CSV file has been processed and saved as", new_metadata_file)
