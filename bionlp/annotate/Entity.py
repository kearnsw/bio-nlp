from bionlp.annotate.Word import Word


class Entity(Word):

    def __init__(self, lexical_form):
        super().__init__(lexical_form)
        self.normal_form = None
