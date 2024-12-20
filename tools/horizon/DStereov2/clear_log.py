results = []
selected = False
with open("/home/fa.fu/work/work_dirs/horizon/DStereov2/20241216/output_v20/hb_mapper_makertbin.log", "r") as file:
    for line in file.readlines():
        if line.startswith("-------------------------------------------------------------------------------------"):
            selected = True
        elif line.startswith("2024"):
            selected = False
        
        if selected:
            if line.split()[-1] == "int8":
                continue
            elif line.split()[0].split('/')[-1].startswith("Conv"):
                continue
            elif selected:
                results.append(line.split()[0] + "\n")

with open("/home/fa.fu/work/work_dirs/horizon/DStereov2/20241216/output_v20/hb_mapper_makertbin.txt", "w") as file:
    for line in results:
        file.write(line)