# CLASSES & OBJECT

class Person:
  # 属性 props
  __name = ''   # 私有属性: 以__开头即为私有属性,在当前类的内部访问
  __email = ''
  age = 20    # 普通属性: 可以在类的内部和外部访问

  # 构造函数
  def __init__(self,name,email):
    self.__name = name
    self.__email = email

  # 方法 methods
  def set_name(self,name):
    self.__name = name
    
  def get_name(self):
    return self.__name

  def set_email(self,email):
    self.__email = email
    
  def get_email(self):
    return self.__email

  def toString(self):
    return '{} is goodlooking, 他的email是{}'.format(self.__name,self.__email)

# # 实例化对象
# henry = Person()

# # 赋值名字
# henry.set_name("Henry")
# # 赋值邮箱
# henry.set_email("27732357@qq.com")

# # 获取名字和邮箱
# print(henry.get_name())
# print(henry.get_email())

bucky = Person("Bucky","Bucky@gmail.com")
# print(bucky.get_name())
# print(bucky.get_email())
# print(bucky.toString())

# 获取私有属性__name
# print(bucky.__name) # 不允许

# 获取普通属性 age
# print(bucky.age) # 允许


# 继承
class Customer(Person):
  __balance = 0

  def __init__(self,name,email,balance):
    self.__name = name
    self.__email = email
    self.__balance = balance
    super(Customer,self).__init__(name,email)

  def set_balance(self,balance):
    self.__balance = balance

  def get_balance(self):
    return self.__balance

  def toString(self):
    # return '{} is goodlooking, 他的email是{}, 他的balance是{}'.format(super(Customer,self).get_name(),super(Customer,self).get_email(),self.__balance)
    return '{} is goodlooking, 他的email是{}, 他的balance是{}'.format(self.get_name(),self.get_email(),self.__balance)



john = Customer("John","john@gmail.com",100)

# 调用父级
# print(john.toString())

# 改名
john.set_name("John Doe")

# 改邮箱
john.set_email("27732357@qq.com")

# 调用重写父级的方法
print(john.toString())