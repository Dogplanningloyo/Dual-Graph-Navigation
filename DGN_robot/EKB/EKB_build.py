import csv
import pickle

def pkl_read(pkl_name):
    with open(pkl_name, "rb") as file:
        pkl_data_get = pickle.load(file)
    return pkl_data_get

def pkl_write(pkl_name,lst_data):
    with open(pkl_name,'wb') as file:
        pickle.dump(lst_data,file)

def csv_read(csv_name,change_float = False):
    with open(csv_name, "r", newline='') as file:
        reader = csv.reader(file)
        if change_float:
            csv_data_get = [float(row[0]) for row in reader]
        else:
            csv_data_get = [row[0] for row in reader]

    return csv_data_get

def csv_write(csv_name,lst_data):
    with open(csv_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for item in lst_data:
            writer.writerow([item])

if __name__ == '__main__':
    print("csv change pickle")
    # with open("all_obj_name.pkl", "rb") as fp:
    #     obj_names = pickle.load(fp)
    #
    # with open("objs_avg_radius.pkl", "rb") as fp:
    #     obj_avg_radius = pickle.load(fp)
    #
    # with open("near_rate_array.pkl", "rb") as fp:
    #     obj_rates = pickle.load(fp)

    # 从EKB中的csv文件读取对应数据
    obj_names = csv_read("all_obj_name.csv")
    obj_avg_radius = csv_read("objs_avg_radius.csv",change_float=True)
    # obj_rates = csv_read("near_rate_array.csv")

    # 将数据写为EKB数据中的二进制文件（提高文件读取速度）
    pkl_write("all_obj_name.pkl",obj_names)
    pkl_write("objs_avg_radius.pkl",obj_avg_radius)
    # pkl_write("near_rate_array.pkl",obj_rates)

    # 从存储的二进制数据中读取数据
    pkl_read("all_obj_name.pkl")
    pkl_read("objs_avg_radius.pkl")
    # pkl_read("near_rate_array.pkl")

    # 重新放入csv文件中，便于二次校验
    csv_write("all_obj_name.csv",obj_names)
    csv_write("objs_avg_radius.csv",obj_avg_radius)
    # csv_write("near_rate_array.csv",obj_rates)

    # with open("all_obj_name.csv",'w',newline='') as fp:
    #     writer = csv.writer(fp)
    #     for item in obj_names:
    #         writer.writerow([item])
    #
    # with open("objs_avg_radius.csv",'w',newline='\n') as fp:
    #     writer = csv.writer(fp)
    #     writer.writerow(obj_avg_radius)
    #
    # with open("near_rate_array.csv",'w',newline='\n') as fp:
    #     writer = csv.writer(fp)
    #     writer.writerow(obj_rates)

    print("EKB Changed")

