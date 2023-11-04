# 输入两个数字相加
num1 = input("请输入数字1：")
num2 = input("请输入数字2：")
sum=int(num1)+int(num2)
print(sum)


# 生成随机数，这个是百度的
import random
random.randint(1,10)

# 九九乘法表格
for i in range(9):
    for j in range(9):
        i
        if i>=j:
            s=(i+1)*(j+1)
            print("%d"%(i+1),"*%d"%(j+1),"=%d"%s,end=" ")
        else:
            print()

