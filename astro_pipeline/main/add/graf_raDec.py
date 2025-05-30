import os
# import matplotlib 
import matplotlib.pyplot as plt 
import mplcursors
# from matplotlib.pyplot import scatter
# matplotlib.use("Agg")

path = "PROCESS_FILE"

ra_val = []
dec_val = []
ra_str_val = []
dec_str_val = [] 
file_names = []


def ra_to_dec(ra):
    h, m, s = map(float, ra.split(":"))
    return (h + m / 60 + s / 3600) * 15


def dec_to_dec(dec):
    d, m ,s = map(float, dec.split(":"))
    if d < 0:
        return d - m / 60 - s / 3600
    return d + m / 60 + s / 3600


def process_file(file_path):
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("#"):
                continue
            try:
                x_str, y_str = line.split()
                ra = ra_to_dec(x_str)
                dec = dec_to_dec(y_str)
                ra_val.append(ra)
                dec_val.append(dec)
                ra_str_val.append(x_str)
                dec_str_val.append(y_str)
                file_names.append(os.path.basename(file_path))
            except ValueError:
                continue
            
for filename in os.listdir(path):
    file_path = os.path.join(path, filename)
    if os.path.isfile(file_path):
        process_file(file_path)
     
print(f"X: {ra_val}\nY: {dec_val}")     
     
if ra_val and dec_val:
    
    unique_files = list(set(file_names))
    
    colors = plt.cm.get_cmap("tab20", len(unique_files))
    
    fig, ax = plt.subplots()
    
    scatter_dir = {}
    inx_dir = {}
    
    # plot_x = []
    # plot_y = []
    
    
    for i, unique_file in enumerate(unique_files):
        ra_file = [ra_val[j] for j in range(len(file_names)) if file_names[j] == unique_file]
        dec_file = [dec_val[j] for j in range(len(file_names)) if file_names[j] == unique_file]
        inx_file = [j for j in range(len(file_names)) if file_names[j] == unique_file]
        
        
        scatter = ax.scatter(ra_file, dec_file, color=colors(i), label=unique_file, s=30, alpha=0.8)
        scatter_dir[unique_file] = scatter
        inx_dir[unique_file] = inx_file
        
        # plot_x.extend(ra_file)
        # plot_y.extend(dec_file)
        
    # plt.scatter(ra_val, dec_val)
    plt.xlabel('X')
    plt.ylabel("Y")
    plt.grid(True)
    
   
    
    # plt.legend(title="Files", loc='center left', bbox_to_anchor=(1, 0.5))
    
    cursor = mplcursors.cursor(scatter_dir.values(), hover=True)
    
    @cursor.connect('add')
    def on_add(sel):
        scatter = sel.artist
        file_name = [name for name, obj in scatter_dir.items() if obj == scatter][0]
        inx = sel.index
        orig_inx = inx_dir[file_name][inx]
        ra_orig = ra_str_val[orig_inx]
        dec_orig = dec_str_val[orig_inx]
        
        # sel.annotation.set_text(f"File: {file_name}\nRA: {sel.target[0]:.6f}\nDEC: {sel.target[1]:.6f}") 
        sel.annotation.set_text(f"File: {file_name}\nRA: {ra_orig}\nDEC: {dec_orig}")    
    
    plt.tight_layout()
    plt.savefig("ra_dec_plot.png", dpi=300)
    plt.show()
    
    with open("ra_dec_values.txt", "w") as f:
        for i in range(len(ra_str_val)):
            f.write(f"{ra_str_val[i]}\t{dec_str_val[i]}\t{file_names[i]}\n")
            
    # with open("x_y_values.txt", "w") as f:
    #     for i in range(len(plot_x)):
    #         f.write(f"{plot_x[i]:.6f}\t{plot_y[i]:.6f}\t{file_names[i]}\n")

else:
    print("Not found")
        
                