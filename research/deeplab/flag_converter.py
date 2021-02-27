class FlagConverter(object):
    FLAGS = []
    def __init__(self):
        self.config = self.FLAGS
    def DEFINE_float(self, name, value, description):
        self.config.append(dict(type="float", name=name, value=value, description=description))
    def DEFINE_boolean(self, name, value, description):
        self.config.append(dict(type="boolean", name=name, value=value, description=description))
    def DEFINE_multi_integer(self, name, value, description):
        self.config.append(dict(type="multi_integer", name=name, value=value, description=description))
    def DEFINE_string(self, name, value, description):
        self.config.append(dict(type="string", name=name, value=value, description=description))
    def DEFINE_enum(self, name, value, l, description):
        self.config.append(dict(type="enum", name=name, value=(value, l), description=description))
    def DEFINE_list(self, name, value, description):
        self.config.append(dict(type="list", name=name, value=value, description=description))
    def DEFINE_integer(self, name, value, description):
        self.config.append(dict(type="integer", name=name, value=value, description=description))
    def DEFINE_multi_float(self, name, value, description):
        self.config.append(dict(type="multi_float", name=name, value=value, description=description))
    def DEFINE_bool(self, name, value, description):
        self.config.append(dict(type="bool", name=name, value=value, description=description))