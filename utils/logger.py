class Logger:
    def __init__(self, log_path: str = 'log.txt', epochs: int = 0, dataset_size: int = 0, components: list = [], float_round: int = -1):
        self.log_path = log_path
        self.epochs = epochs
        self.dataset_size = dataset_size

        self.components = []
        self.float_round = -1

        self.set_float_round(float_round)
        self.set_sort(components)

        self.current_line = ''
        self.current_line_components = []
        self.wf = open(self.log_path, 'w')

    def __del__(self):
        self.wf.close()

    def set_sort(self, components: list):
        self.components = components

    def set_float_round(self, float_round: int):
        self.float_round = float_round

    def _add_component(self, key, data):
        if isinstance(data, list) and len(data) == 2:
            data = data[0] / data[1] if data[1] != 0 else 0
        if isinstance(data, float) and self.float_round > 0:
            data = round(data, self.float_round)
        self.current_line_components.append('{}: {}'.format(key, data))

    def _make_line(self, titles, comp_dict):
        self.current_line_components = []
        titles = list(titles)

        title = ''
        if 'epoch' in comp_dict.keys():
            titles = ['[{}/{}]'.format(comp_dict['epoch'], self.epochs)] + titles
        if len(titles) > 0:
            title += '{}'.format(' '.join(titles))
        if 'batch' in comp_dict.keys():
            title += '[{}/{}]'.format(comp_dict['batch'], self.dataset_size)
        if len(title) > 0:
            self.current_line_components.append(title)
        if 'components_data' in comp_dict.keys():
            assert(isinstance(comp_dict['components_data'], dict))
            assert(len(comp_dict['components_data']) == len(self.components))
            for key, data in comp_dict['components_data'].items():
                self._add_component(key, data)

        for key in self.components:
            if key in comp_dict.keys():
                self._add_component(key, comp_dict[key])

        for key in sorted(comp_dict.keys()):
            if key not in self.components and key not in ['epoch', 'batch', 'components_data']:
                self._add_component(key, comp_dict[key])

        self.current_line = '  '.join(self.current_line_components)

    def write_log(self, *titles, **kwargs):
        self._make_line(titles, kwargs)
        self.wf.write(self.current_line + '\n')

    def print_log(self, *titles, **kwargs):
        self._make_line(titles, kwargs)
        print(self.current_line)

    def print_and_write_log(self, *titles, **kwargs):
        self._make_line(titles, kwargs)
        self.wf.write(self.current_line + '\n')
        print(self.current_line)

    def __call__(self, *titles, **kwargs):
        self._make_line(titles, kwargs)
        self.wf.write(self.current_line + '\n')
        print(self.current_line)
