from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def report_nb(df,model) :
    data_latih, data_uji = train_test_split(df, test_size=0.2, random_state=42)
    predictions = model.predict(data_latih, data_uji)
    print(classification_report(data_uji[data_uji.columns[-1]], predictions))

def report_knn(df,model) :
    data_latih, data_uji = train_test_split(df, test_size=0.2, random_state=42)
    predictions = model.predict(data_latih, data_uji)
    print(classification_report(data_uji[data_uji.columns[-1]], predictions))

def report_svm(df,model) :
    data_latih, data_uji = train_test_split(df, test_size=0.2, random_state=42)
    model.fit(data_latih, data_latih[data_latih.columns[-1]])
    predictions = model.predict(data_uji)
    print(classification_report(data_uji[data_uji.columns[-1]], predictions))

def report_dt(df,model):
    data_latih, data_uji = train_test_split(df, test_size=0.2, random_state=42)
    model.fit(data_latih, data_latih.columns[-1])
    data_uji_dict = data_uji.iloc[:,:-1].to_dict(orient = "records")
    predictions = []
    for i in range(len(data_uji_dict)):
        predict = model.predict(data_uji_dict[i])
        predictions.append(predict)
    print(classification_report(data_uji[data_uji.columns[-1]], predictions))