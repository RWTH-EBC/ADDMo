
NameOfData = "sin(x1)"

#save dataframe in an excel file
ExcelFile = "%s\\ArtificialData_%s.xlsx" % (ResultsFolder, NameOfData)
writer = pd.ExcelWriter(ExcelFile)
Data.to_excel(writer, sheet_name="ImportData")
writer.save()
writer.close()