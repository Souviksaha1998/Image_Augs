import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
import datetime
from rich.console import Console

console  = Console()
warnings.filterwarnings("ignore")

class DataAnalyser:
    def __init__(self, source_path,class_path=None,is_json=False):
     
        self.source_path = source_path
        
        self.is_json = is_json
  
        if class_path is None:
            if not self.is_json:
                self.class_path = os.path.join(source_path, 'classes.txt')
                if not os.path.exists(self.class_path):
                    raise ("Class path does not exsist")
 
                elif not os.path.isfile(self.class_path):
                        raise ("Class path must be a file")
    
        self.data = {}

        if not os.path.exists(self.source_path):
            raise ("Source path does not exsit")
        elif not os.path.isdir(self.source_path):
            raise ("Source path must be a directory")
        
        os.makedirs('plots' , exist_ok=True)


        
    
    def read_txt_file(self, path):

        if self.is_json:
            with open(path, 'r') as files:
                datas = json.load(files)['shapes']
            
            return datas
        
        else:
            with open(path, 'r') as f:
                contents =  f.readlines()

            contents = [i.replace("\n", "").strip() for i in contents if i != "\n"]
            return contents

    def plot_graph(self,save_folder):
        # current_time = str(datetime.datetime.now().strftime("%H:%M:%S")).split(':')
        current_time = save_folder
        labels = list(self.data.keys())
        values = list(self.data.values())
        # Create subplots with two columns (for pie chart and bar plot)
        fig, axes = plt.subplots(1, 2, figsize=(18,8))

    
        # Create and save the pie chart
        axes[0].pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
        axes[0].axis('equal')  
        axes[0].set_title('Pie Chart')

        
        sns.barplot(x=labels, y=values, ax=axes[1])
        axes[1].set_title('Bar Plot')

        for i, (label, value) in enumerate(zip(labels, values)):
            axes[1].text(i, value + 1, f"{value}", ha='center', va='bottom')
        
        plt.xticks(rotation=90)
        plt.tight_layout()

        
        plt.savefig(f'plots/{current_time}.png',dpi=200)

        # plt.show()

        # console.print(f'[bold green] [+] labels distribution plot "labels_{"_".join(current_time)}.png" generated successfully in plots folder [bold green]')

    def analyse(self,save_folder):
        
        if not self.is_json:
          class_labels = self.read_txt_file(self.class_path)

        for file in os.listdir(self.source_path):
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                if self.is_json:
                    txt_path = os.path.join(self.source_path, f"{'.'.join(file.split('.')[:-1])}.json")    
                else:
                    txt_path = os.path.join(self.source_path, f"{'.'.join(file.split('.')[:-1])}.txt")
                
                if os.path.exists(txt_path):
                    contents = self.read_txt_file(txt_path)
                    

                    if self.is_json:
                        for content in contents:
                            if content['label'] not in self.data:
                                
                                self.data[content['label']] = 1
                            else:
                                self.data[content['label']] += 1
                
                    else:
                        for content in contents:
                        
                            class_name = class_labels[int(content.split()[0])]
                        
                            if class_name not in self.data:
                                self.data[class_name] = 1
                            else:
                                self.data[class_name] += 1
        
        self.plot_graph(save_folder)


if __name__ == '__main__':

    analyser = DataAnalyser("test_imges\object_detection",is_json=False)

    analyser.analyse()