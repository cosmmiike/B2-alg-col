import numpy as np
import random
import matplotlib.pyplot as plt


# fonction à optimiser
fonc = lambda x: - x ** 2 * (2 + np.sin(10 * x)) ** 2
val_atteint = 0

derive = lambda x: - 2 * x * (2 + np.sin(10 * x)) ** 2 - 20 * x ** 2 * (2 + np.sin(10 * x)) * np.cos(10 * x)


def tracerFonction(limiteGauche=-1,
                   limiteDroite=1):
    x = np.arange(limiteGauche, limiteDroite+0.01, 0.01)
    y = fonc(x)
    plt.plot(x, y)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.title(r'$f(x)=-(x^2(2+sin(10x))^2)$')
    plt.grid(True)
    plt.show()


class Individus:
    def __init__(self,
                 limiteGauche=-1,
                 limiteDroite=1):
        self.limiteGauche = limiteGauche
        self.limiteDroite = limiteDroite
        self.x = random.uniform(self.limiteGauche, self.limiteDroite)
        self.y = fonc(self.x)
        self.fit = 0

    def fitness(self,
                val_att=val_atteint):
        self.fit = val_att - self.y
        return self.fit

    def info(self):
        self.fitness()
        print(self.x, end='\t\t')
        print(self.y, end='\t\t')
        print(self.fit)


class AlgorithmeGenetique():
    def __init__(self,
                 population_taille=10,
                 prop_grd_mutation=.8,
                 prop_pte_mutation=.0,
                 prop_sexe=.0,
                 precision=1e-8):
        self.population_taille = population_taille
        self.prop_grd_mutation = prop_grd_mutation
        self.prop_pte_mutation = prop_pte_mutation
        self.prop_sexe = prop_sexe
        self.precision = precision
        self.population = []
        for i in range(self.population_taille):
            nouvel_ind = Individus()
            self.population.append(nouvel_ind)

    def sort_population(self):
        self.population.sort(key=lambda param: param.y, reverse=False)


    def info_population(self):
        # for i in range(self.population_taille):
        #     self.population[i].fitness()
        #     print("{0:10.3E}".format(self.population[i].x), end='\t\t')
        #     print("{0:10.3E}".format(self.population[i].y), end='\t\t')
        #     print("{0:10.3E}".format(self.population[i].fit))
        self.population[-1].fitness()
        print("{0:10.3E}".format(self.population[-1].x), end='\t\t')
        print("{0:10.3E}".format(self.population[-1].y), end='\t\t')
        print("{0:10.3E}".format(self.population[-1].fit))

    def grande_mutation(self,
                        Individus):
        # print()
        # Individus.info()
        self.individus = Individus
        self.individus.x = random.uniform(self.individus.limiteGauche, self.individus.limiteDroite)
        self.individus.y = fonc(self.individus.x)
        self.individus.fitness()
        if self.individus.fit < Individus.fit:
            Individus.x = self.individus.x
            Individus.y = self.individus.y
            Individus.fit = self.individus.fit

        # Individus.info()
        # print()

    def petite_mutation(self,
                        Individus,
                        mutation_taille=.1):
        self.mutation_taille = mutation_taille
        # print()
        # Individus.info()
        individus_max = self.population[-1]


        # ПОЧему self.individus влияет на Individus ???????

        self.individus = Individus
        self.individus.x = individus_max.x + random.uniform(-self.mutation_taille, self.mutation_taille)
        self.individus.y = fonc(self.individus.x)
        self.individus.fitness()
        # if self.individus.fit < Individus.fit:
        #     Individus.x = self.individus.x
        #     Individus.y = self.individus.y
        #     Individus.fit = self.individus.fit
        # Individus.info()
        # print()
        return Individus

    def crossover(self,
             Individus1,
             Individus2):
        # print()
        # Individus1.info()
        moyen_x = derive((Individus1.x + Individus2.x)/2)
        if derive(moyen_x) <= 0:
            Individus2.x = (Individus1.x + Individus2.x) / 2
            Individus2.y = fonc(Individus2.x)
            Individus2.fitness()
            # Individus2.info()
        else:
            Individus1.x = (Individus1.x + Individus2.x)/2
            Individus1.y = fonc(Individus2.x)
            Individus1.fitness()
            # Individus1.info()
        # print()

    def nouvelleGeneration(self):
        nb_grande_mutation = round(self.population_taille * self.prop_grd_mutation)
        nb_petite_mutation = round(self.population_taille * self.prop_pte_mutation)
        nb_crossover = round(self.population_taille * self.prop_sexe)
        # for i in range (0, nb_grande_mutation):
        #     self.grande_mutation(self.population[i])
        for i in range (-1, self.population_taille - nb_petite_mutation):
            self.petite_mutation(self.population[i])
        # for i in range (nb_grande_mutation, self.population_taille - nb_petite_mutation):
        #     self.crossover(self.population[i], self.population[i+1])


        return self


def run():
    # tracerFonction(-1,1)
    ag = AlgorithmeGenetique(10, .8,)
    ag.sort_population()
    ag.info_population()
    print()
    for i in range (100):
        ag.nouvelleGeneration()
        ag.sort_population()
        ag.info_population()
        print()


if __name__ == "__main__":
    run()