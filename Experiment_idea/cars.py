from genetic import *
import numpy as np



class MyCar(Car):
    def __init__(self, chromosome, **kwargs):
        super(MyCar, self).__init__(**kwargs)

        self.input_dim = len(self.sensors) + 2  # two additional inputs for the current speed and the current turn
        self.output_dim = 2  # there are 2 options: 1, go left/right and 2, go forward/backward
        
        ##  ----- MY CODE  ----
        ## definujme si dimenzie a aktiavacne funckie
        ## iba 2 vahove matice
        self.dimensions = [self.input_dim, 64, self.output_dim]
        activation_functions = [self.sigmoid, self.tanh]

        ## 3 vahove maticevahove matice
        # self.dimensions = [self.input_dim, 8, 8, self.output_dim]
        # activation_functions = [self.sigmoid, self.sigmoid, self.tanh]
        
        ## kludne aj 4 vahove vahove matice
        # self.dimensions = [self.input_dim, 10, 10, 10, self.output_dim]
        # activation_functions = [self.sigmoid, self.sigmoid, self.sigmoid, self.tanh]
        
        chromosome_lenght, splitting_points = self.get_chromosome_lenght_and_splitting_points()

        
        if chromosome is None:
            # # # YOUR CODE GOES HERE # # #
            # create the initial values for a chromosome -> list of random numbers
            self.chromosome = list(np.random.uniform(-1, 1, chromosome_lenght))
        else:
            self.chromosome = chromosome

        # # # YOUR CODE GOES HERE # # #
        # create at least two numpy matrices from self.chromosome, which you will then use to control the car with
        # tip: try to use numpy.reshape(vector) to change a vector dimensions!

        """
        # TOTO TREBA AUTOMATIZOVAT
        # idea: nemajme W_in, W_out (W_in, W_middle, W_out pre bonus) ale spravme si zoznam 
        # váh pre neuronku, ktorý bude osahovať matice váh každej vrstvy + smerník na aktivačnú funkciu
        # sice to vyzerá byť ako práca navyše, ale riešenie je rozbustnejšie a ľohko pridám 3, 4, 5 .. skrytých vrstiev

        in_vector = self.chromosome[:self.hidden_dim * (self.input_dim + 1)]
        self.W_in = np.reshape(in_vector, (self.hidden_dim, (self.input_dim + 1)))

        # mozno tu pojde este nejaka (bonus)

        out_vector = self.chromosome[self.hidden_dim * (self.input_dim + 1):]
        self.W_out = np.reshape(out_vector, (self.output_dim, (self.hidden_dim + 1)))
        """
        
        flat_arrays = np.split(self.chromosome, splitting_points)

        self.W_array = list()
    
        for i, flat_array in enumerate(flat_arrays):
            input_dimension = self.dimensions[i]
            output_dimension = self.dimensions[i + 1]
            matrix_and_activation = (np.reshape(flat_array, (output_dimension, (input_dimension + 1))), activation_functions[i])
            self.W_array.append(matrix_and_activation)

        # self.W_in, self.W_hidden_1, self.W_hidden_2, self.W_out = tuple(self.W_array) ## toto netreba, iba som chcel ukazat ako by som s maticami pracoval, budem to robiť však v cykle

    def get_chromosome_lenght_and_splitting_points(self):
        # do chromozómu chceme zokódovať všetky matice váh a niekedy v budúcnosti ich budeme chcieť rozbaliť 
        # na jednotlivé vrstvy, na to slúži splitting_points  
        chromosome_lenght = 0
        splitting_points = []
        for i in range(1, len(self.dimensions)):
            input_dimension = self.dimensions[i - 1]
            output_dimension = self.dimensions[i]
            chromosome_lenght += output_dimension * (input_dimension + 1)
            splitting_points.append(chromosome_lenght)
        return chromosome_lenght, splitting_points[:-1]

    def compute_output(self):
        sensor_norms = []
        # # # YOUR CODE GOES HERE # # #
        # create an array of sensor lengths. Afterwards, append to this array the speed and the turn of the car
        # (self.speed and self.turn). This array serves as an input to the network (but do not forget the bias)
        # the sensor lengths values are originally quite large - are in the interval <0, 200>. To prevent "explosion" in
        # the neural network, normalize the lengths to smaller interval, e.g. <0, 10>

        for point_1, point_2 in self.sensors:
            sensor_norms.append(min(200, np.linalg.norm(point_2 - point_1)) / 20.0) ## Prečo nenormalizovať medzi <0, 1> ?
        sensor_norms.append(self.speed)
        sensor_norms.append(self.turn)
    
    
        # # # YOUR CODE GOES HERE # # #
        # use the input to calculate the instructions for the car at the given state
        # do not forget to add bias neuron before each projection!
        # for adding bias you can use np.concatenate(( state , [1]))
        # between two layers use an activation function. sigmoid(x) is already implemented, however, feel free to use
        # other, meaningful activation function
        # at the end of the network, use hyperbolic tangent -> to ensure that the change is in the range (-1, 1)

        state = np.array(sensor_norms)
        for W, f in self.W_array:
            state = np.append(state, 1)
            state = f( np.dot(W, state) ) 
        return state
    
    def sigmoid(self, x):
        # Activation function - logistical sigmoid
        return 1 / (1 + np.e  ** -x)
    
    def tanh(self, x):
        # Activation function - hyperbolic tangens
        e = np.e
        return (e ** x - e ** - x) / (e ** x + e ** - x)
    
    def leaky_relu(self, x):
        # Activation function - leaky relu
        return np.maximum(0.1 * x, x)
    
    def update_parameters(self, instructions):
        # # # YOUR CODE GOES HERE # # #
        # use the values outputted from the neural network (instructions) to change the speed and turn of the car
        self.speed += instructions[0] * self.max_speed_change
        self.turn += instructions[1] * self.max_turn_change

        self.speed = np.clip(self.speed, self.speed_limit[0], self.speed_limit[1])
        self.turn = np.clip(self.turn, self.turn_limit[0], self.turn_limit[1])
        self.orientation += self.turn


class MyRaceCars(RaceCars):
    def __init__(self, *args, **kwargs):
        super(MyRaceCars, self).__init__(*args, **kwargs)

    def generate_individual(self, chromosome=None):
        initial_position = [500, 600 + random.randint(-25, 25)]
        orientation = 90
        new_car = MyCar(chromosome, position=initial_position, orientation=orientation,
                        additional_scale=self.scale_factor)
        sensors = []
        if self.show_sens:
            for _ in new_car.sensors:
                sensors.append(self.canvas.create_line(0, 0, 0, 0, dash=(2, 1)))
        return [new_car, self.canvas.create_image(initial_position), sensors]


if __name__ == '__main__':
    track = create_track()

    # useful parameters to play with:
    population_size = 32        # total population size used for training
    select_top = 4              # during selection, only the best select_top cars are chosen as parents

    show_training = True        # each generation is shown on canvas
    show_all_cars = False       # the code is faster if not all cars are always displayed
    displayed_cars = 8          # only the first few cars are displayed

    # show_training = False     # the training is done in the background
    show_every_n = 10            # the best cars are shown after every few generations (due to faster training)

    mutation_prob = 0.05        # mutation probability for number mutation
    deviation = 1               # this standard deviation used when mutating a chromosome

    RC = MyRaceCars(track, population_size=population_size, show_sensors=False, gen_steps=500, n_gens=100,
                    show_all_cars=show_all_cars, select_top=select_top, mutation_prob=mutation_prob,
                    show_training=show_training, displayed_cars=displayed_cars, vis_pause=10, show_every_n=show_every_n,
                    can_width=1100, deviation=deviation)

    RC.run_simulation()
