import csv




class DataKeeper(object):
    def __init__(self,filename='data.csv',fields=[]) -> None:
        self.filename = filename
        self.__field = fields
        try:
            self.__csvFile = open(filename,'a',newline='')
        except:
            print ('cannot open the file ')
        self.__writer = csv.DictWriter(self.__csvFile,fieldnames=fields)
        

    def addRow(self,row):
        for i in self.__field:
            if i not in row:
                return False
        print (row)
        self.__writer.writerow(row)
        self.__csvFile.flush()
        

