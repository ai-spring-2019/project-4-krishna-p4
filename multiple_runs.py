import csv

with open("run_control.csv", "r") as file:
	delimiter = ","
	reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
        	for cell in line:
        		


