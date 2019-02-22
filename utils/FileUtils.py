import csv


class FileUtils(object):

    @staticmethod
    def save_dictionary(tuple_to_save, file, mode='w'):
        with open(file, mode, newline='') as csv_file:
            spam_writer = csv.writer(csv_file)
            for key, value in tuple_to_save.items():
                spam_writer.writerow([key, "{}".format(value)])

    @staticmethod
    def read_dictionary(file) -> {}:
        result = {}
        with open(file, newline='') as csv_file:
            spam_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
            for row in spam_reader:
                result[row[0]] = row[1]
        return result
