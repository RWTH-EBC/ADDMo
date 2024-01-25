import os
NameOfData = "sin(x1)"

#save dataframe in an excel file
ExcelFile = os.path.join(ResultsFolder, "ArtificialData_%s.xlsx"%(NameOfData))
writer = pd.ExcelWriter(ExcelFile)
Data.to_excel(writer, sheet_name="ImportData")
writer.save()
writer.close()