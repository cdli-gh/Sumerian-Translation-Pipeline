import random
lines=[]
with open("CDLI_Data/Sumerian_monolingual_processed.txt", "r") as f:
    for line in f:
        line=line.strip()
        lines.append(line)

def savefile(filename,LIST):
    with open(filename, 'w') as f:
        for line in LIST:
            f.write("%s\n" % line)
        
        
        
random.seed(12)
random.shuffle(lines)
demo=lines[0:150]
print("150 Random Sumerian phrases (out of 1.5M) saved at Dataset/sumerian_demo.txt ")
savefile("Dataset/sumerian_demo.txt",demo)
