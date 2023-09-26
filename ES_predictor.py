import numpy as np
import pandas as pd
import joblib
import argparse


def main():
    parser = argparse.ArgumentParser(description="ES_predictor")

    parser.add_argument('--input', '-i', required=True, help='input file')
    parser.add_argument('--output', '-o', required=True, help='output file')
    parser.add_argument('--model', '-m', required=True, help='best model')

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    model = args.model

    data = pd.read_excel(input_file)  #reading data
    #print(data)
    Data=np.array(data)

    data_ML=np.stack(Data)[:,3:9]
    #print(data_ML)
    data_info=np.stack(Data)[:,0:3]
    #print(data_info)

    forest=joblib.load(model)
    predict=forest.predict(data_ML)

    Data_new=np.c_[data_info,predict]
    new_Data=pd.DataFrame(Data_new)
    new_Data.columns = ["sample","latitude","longitude", "predicted ES"]
    #print(new_Data)

    new_Data.to_excel(output_file,index=False)


if __name__ == "__main__":
    main()