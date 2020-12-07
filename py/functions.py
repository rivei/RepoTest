# FUNCTIONS

# Create a function
def sayHello(name = "Henry"):  # 函数声明
  # 代码块
  print("Hello",name)

# 函数调用
# sayHello()


# Return a Value
def getSum(num1,num2):
  total = num1 + num2
  return total
  # print(123)

sumValue = getSum(1,2)
# print(sumValue)

# 作用域

# 在函数内的变量,出了函数,作用域就消失了
# number = 2

# def numberScope():
#   number = 200
#   print(number)  # 200

# numberScope()
# print(number)  # 2

# 两种不同的数据,在函数中会出现不同的结果
def addOneToNum(num):
  num = num + 1
  print("Value inside function:", num)
  return

num = 5
# addOneToNum(num)
# print("Value outside function:", num)

def addOneToList(myList):
  myList.append(4)
  print("Value inside function:", myList)
  return 

myList = [1,2,3]
addOneToList(myList)
print("Value outside function:",myList)

