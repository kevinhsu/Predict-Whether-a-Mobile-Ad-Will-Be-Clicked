from csv import DictReader
import csv

train = './data/train.csv'
test = './data/test.csv'

#You can live out any one of the variables if you want simple by using #
def data(path):
    for t, row in enumerate(DictReader(open(path))):
        id = row['id']
        # click = int(row['click'])   # For training set
        hour = int(row['hour'])
        C1 = int(row['C1'])
        banner_pos = int(row['banner_pos'])
        site_id = row['site_id']
        site_domain = row['site_domain']
        site_category = row['site_category']
        app_id = row['app_id']
        app_domain = row['app_domain']
        app_category = row['app_category']
        # device_id = row['device_id']
        # device_ip = row['device_ip']
        device_model = row['device_model']
        device_type = int(row['device_type'])
        device_conn_type = int(row['device_conn_type'])
        C14 = int(row['C14'])
        C15 = int(row['C15'])
        C16 = int(row['C16'])
        C17 = int(row['C17'])
        C18 = int(row['C18'])
        C19 = int(row['C19'])
        C20 = int(row['C20'])
        C21 = int(row['C21'])

        # Set weekday and day_time as new features
        if (hour >= 14102400 and hour <= 14102423) or (hour >= 14103100 and hour <= 14103123):
            friday = 1
        else:
            friday = 0

        # Time here is meaningless for training, it should transform to Friday and non-Friday
        day_time = hour % 100

        # x = [id,click,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21,friday,day_time]
        x = [id,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21,friday, day_time]

        yield t, x

# x_names = ["id","click","hour","C1","banner_pos","site_id","site_domain","site_category","app_id","app_domain","app_category","device_id","device_ip","device_model","device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21","friday","day_time"]
x_names = ["id","hour","C1","banner_pos","site_id","site_domain","site_category","app_id","app_domain","app_category","device_model","device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21","friday","day_time"]


# train_out_0 = './data/train_out_0.csv'
# train_out_1 = './data/train_out_1.csv'
train_out = './data/train_out2.csv'
test_out = './data/test_out2.csv'

# csvfile_out = open(train_out_0, 'w')
# data_out = csv.writer(csvfile_out)
# data_out.writerow(x_names)
# for t, x in data(train):
#     if x[1] == 0:
#         data_out.writerow(x)
# csvfile_out.close()
# del data_out
# print "%s is ready." % (train_out_0)
#
# csvfile_out = open(train_out, 'w')
# data_out = csv.writer(csvfile_out)
# data_out.writerow(x_names)
# for t, x in data(train):
#     data_out.writerow(x)
# csvfile_out.close()
# del data_out
# print "%s is ready." % (train_out)

csvfile_out = open(test_out, 'w')
data_out = csv.writer(csvfile_out)
data_out.writerow(x_names)
for t, x in data(test):
    data_out.writerow(x)
csvfile_out.close()
del data_out
print "%s is ready." % (test_out)



