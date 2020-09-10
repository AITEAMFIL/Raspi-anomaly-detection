import csv
def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r') as infile:
        with open(csv_file, 'w') as outfile:
            stripped = (line.strip() for line in infile)
            lines = (line.split(",") for line in stripped if line)
            writer = csv.writer(outfile)
            writer.writerows(lines)


def csv_to_txt(txt_file, csv_file):
    with open(txt_file, "w") as my_output_file:
        with open(csv_file, "r") as my_input_file:
            # i = 0
            # for row in csv.reader(my_input_file):
            #     i += 1
            #     print(len(row))
            # print(i)
            [ my_output_file.write(",".join(row)+'\n') for row in csv.reader(my_input_file)]
        my_output_file.close()
def mlp_to_txt(ran):
    for i in range(ran):
        bias = 'mlpbias{0}{1}'.format(i + 1, i + 2)
        coefs = 'mlpcoefs{0}{1}'.format(i + 1, i + 2)
        csv_to_txt(bias + '.txt', bias + '.csv')
        csv_to_txt(coefs + '.txt', coefs + '.csv')
def mlp_to_csv(ran):
    for i in range(ran):
        bias = 'mlpbias{0}{1}'.format(i + 1, i + 2)
        coefs = 'mlpcoefs{0}{1}'.format(i + 1, i + 2)
        txt_to_csv(bias + '.txt', bias + '.csv')
        txt_to_csv(coefs + '.txt', coefs + '.csv')
