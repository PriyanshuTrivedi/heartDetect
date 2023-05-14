import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import train_test

df=pd.read_csv("heart.csv")
x_train=train_test.x_train
x_test=train_test.x_test
y_train=train_test.y_train
y_test=train_test.y_test

def train_test_grphs(check_of):
    if check_of=='train':
        val=y_train
    elif check_of=='test':
        val=y_test
    freq_list=(pd.Index(val)).value_counts()
    order=freq_list.index
    plt.figure(figsize=(12,7))
    plt.suptitle(f'Heart Disease distribution on {check_of}ing dataset')
    # --- Pie Chart ---
    plt.subplot(1, 2, 1)
    plt.pie(freq_list,labels=['Positive','Negative'],autopct='%.2f%%',pctdistance=0.7, textprops={'fontsize':9})
    centre=plt.Circle((0, 0), 0.4, fc='white')
    plt.gcf().gca().add_artist(centre)
    # --- Bar Chart ---
    plt.subplot(1, 2, 2)
    plt.bar(['Positive','Negative'],freq_list)
    plt.ylabel('Total')
    plt.xlabel('Heart Disease Status')
    plt.show()

# plot bar and pie charts on non-continuos data
def pie_and_bar_charts(value,labls,xlabl,title,xtck):
    order=df[value].value_counts().index
    plt.figure(figsize=(12,7))
    plt.suptitle(title)
    # --- Pie Chart ---
    plt.subplot(1, 2, 1)
    plt.title('Pie Chart')
    plt.pie(df[value].value_counts(), labels=labls, autopct='%.2f%%',pctdistance=0.7, textprops={'fontsize':9})
    centre=plt.Circle((0, 0), 0.4, fc='white')
    plt.gcf().gca().add_artist(centre)
    # --- Histogram ---
    plt.subplot(1, 2, 2)
    plt.title('Histogram')
    ax=sns.countplot(x=value, data=df,order=order)
    for rect in ax.patches:
        ax.text (rect.get_x()+rect.get_width()/2,rect.get_height()+4.25,rect.get_height(),horizontalalignment='center', fontsize=9)
    plt.xlabel(xlabl)
    plt.ylabel('Total')
    plt.xticks(xtck,labls)
    plt.show()

# plot hitogram of continuous data
def plot_histogram(value,xlabl,title):
    sns.histplot(data=df, x=value, kde=True)
    plt.ylabel('Total')
    plt.xlabel(xlabl)
    plt.title(title)
    plt.show()


def pie_and_bar_charts_of_non_descriptive():
    pie_and_bar_charts('sex',['Female', 'Male'],'Gender','Sex(Gender) Distribution',[0,1])
    pie_and_bar_charts('cp',['Type 0', 'Type 2', 'Type 1', 'Type 3'],'Pain Type','Chest Pain type Distribution',[0,1,2,3])
    pie_and_bar_charts('fbs',['< 120 mg/dl', '> 120 mg/dl'],'Fasting Blood Sugar','Fasting Blood Sugar Distribution',[0,1])
    pie_and_bar_charts('restecg',['1','0','2'],'Resting Electrocardiographic','Resting Electrocardiographic Distribution',[0,1,2])
    pie_and_bar_charts('exang',['False', 'True'],'Exercise Induced Angina','Exercise Induced Angina Distribution',[0,1])
    pie_and_bar_charts('slope',['2', '1', '0'],'Slope','Slope Distribution',[0,1,2])
    pie_and_bar_charts('ca',['0', '1', '2', '3', '4'],'Number of Major Vessels','Number of Major Vessels Distribution',[0,1,2,3,4])
    pie_and_bar_charts('thal',['2', '3', '1', '0'],'Number of "thal"','"thal" distribution',[0,1,2,3])
    pie_and_bar_charts('target',['True','False'],'Heart disease Status','Heart disease distribution',[0,1])

def pie_and_bar_charts_of_descriptive():
    plot_histogram('age','Age','Age Distribution')
    plot_histogram('trestbps','Resting Blood Pressure','Resting Blood Pressure Distribution')
    plot_histogram('chol','Cholestrol','Cholestrol Distribution')
    plot_histogram('thalach','Max heart rate','Max heart rate Distribution')
    plot_histogram('oldpeak','oldpeak','"oldpeak" Distribution')

def pie_and_bar_after_splitting():
    train_test_grphs('train')
    train_test_grphs('test')

def draw_all_graphs_and_charts():
    pie_and_bar_charts_of_non_descriptive()
    pie_and_bar_charts_of_descriptive()
    pie_and_bar_after_splitting()

draw_all_graphs_and_charts()