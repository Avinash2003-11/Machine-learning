import pandas as pd
from datetime import datetime
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}


df = pd.DataFrame(data)


print(df)

df.head()
print(df)


today_date = datetime.now()

df = pd.read_csv("./data/superstore.csv")

