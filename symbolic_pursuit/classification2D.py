from pysymbolic.models.special_functions import MeijerG
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
from itertools import combinations 
from scipy.optimize import minimize
from scipy.special import softmax
from sympy import Symbol, sympify
import mpmath
import numpy as np

EPS = np.finfo(np.float32).eps
# Hyperparameters related functions:

def load_h():
    h = {
        'hyper_1': (np.array([0.5, 0.0]),
                    [1, 0, 0, 2]),  # Parent of sin , cos, sh, ch
        'hyper_2': (np.array([2.0, 2.0, 2.0,
                              1.0]),
                    [0, 1, 3, 1]),  # Parent of monomials
        'hyper_3': (np.array([0.3, 0.1,
                              0.1, 0.0, 0.3]),
                    [2, 1, 2, 3]),  # Parent of exp, Gamma, gamma

    }
    '''   
        'hyper_4': (np.array([1.0, 1.2, 3.0,
                              3.3, 0.4, 1.5]),
                    [2, 2, 3, 3]),  # Parent of ln, arcsin, arctg
        'hyper_5': (np.array([1.1,
                              1.1, 0.5, -0.5]),
                    [2, 0, 1, 3])  # Parent of Bessel functions
    '''

    return h


# Miscellaneous functions


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return x * (x > 0)


"""
------------------------------------------------------------
Class for symbolic models associated to a regression problem
------------------------------------------------------------
"""


class SymbolicClassifier2D:

    # Adaptation of existing methods:

    def __init__(self, verbosity=True, 
                        loss_tol=1.0e-3, 
                        ratio_tol=0.9,
                        nclasses=10, 
                        maxiter=100,
                        eps=1.0e-5, 
                        random_seed=42):
        self.dim_x = [0, 0]  # Number of features
        self.n_points = 0  # Number of points
        self.alpha = 1. # residual weightage
        self.current_resi = np.array(self.n_points)  # Array of residues
        self.loss_list = []  # List of residual losses associated to each term
        self.terms_list = []  # List of all the terms in the model
        self.random_seed = random_seed  # Random seed for reproducibility
        self.verbosity = verbosity  # True if the optimization process should be detailed
        self.ratio_tol = ratio_tol  # A new term is added only if new_loss / old_loss < ratio_tol
        self.loss_tol = loss_tol  # The tolerance for the loss under which the pursuit stops
        self.maxiter = maxiter  # Maximum number of iterations for optimization
        self.nclasses = nclasses
        self.eps = eps  # Small number used for numerical stability
        if self.verbosity:
            print('Model created with the following hyperparameters :'
                  + '\n loss_tol={} \n ratio_tol={} '
                    '\n maxiter={} \n eps={} \n random_seed={}'.format(loss_tol, ratio_tol, maxiter, eps, random_seed))


    def __str__(self):
        expression = ""
        expression += str(self.get_expression())
        return expression


    # Optimizer

    def optimize_CG(self, loss, theta_0, linear_constraint):
        # Encodes the parameters of the optimal parameters 
        # of an additional term inside theta_opt
        minimizer_kwargs = {'method': 'CG',
                            # 'jac' : '2-point',
                            # 'hess': '2-point',

                            'options': {'disp': self.verbosity,
                                        'gtol': 1e-2,
                                        'maxiter': self.maxiter,
                                        'eps': 0.1,
                                        },
                            'constraints': [linear_constraint]}
        opt = basinhopping(loss, theta_0, 
                            minimizer_kwargs=minimizer_kwargs,
                            niter=self.maxiter,
                            stepsize=1.,
                            niter_success= 3,
                            disp = self.verbosity)
        # opt = minimize(loss, theta_0, method='CG',
        #                constraints=[linear_constraint],
        #                options={'disp': self.verbosity, 
        #                         'maxiter': self.maxiter})
        theta_opt = opt.x
        loss_ = opt.fun
        return theta_opt, loss_

    def forward(self, X, V, U):
        D = np.abs([np.max(np.linalg.eig(x)[0]) for x in X])
        vec = ReLU(np.matmul(np.matmul(X, V), U)
                    / (np.sqrt(D) * np.linalg.norm(V) * np.linalg.norm(U)))
        return vec

    # Extract information from the model

    def predict(self, X, exclude_term=False, exclusion_id=0):
        # Returns the evaluation of the model minus term # exclusion_id at the point in X
        result = np.zeros((len(X), self.nclasses))
        index_list = [k for k in range(len(self.terms_list))]

        if exclude_term:
            index_list.pop(exclusion_id)
        
        for k in index_list:
            meijer_g, vs, w = self.terms_list[k]
            vs = [vs[i*self.dim_x[0]: (i+1)*self.dim_x[0]] for i in range(self.nclasses)]
            offset = self.nclasses*self.dim_x[0]
            us = [vs[offset + i*self.dim_x[1]: offset + (i+1)*self.dim_x[1]] for i in range(self.nclasses)]
            for ii in range(self.nclasses):
                result[:,ii] = result[:,ii] + w * meijer_g.evaluate(self.forward(X, vs[ii], us[ii]))
        return softmax(result, 1)


    def get_expression(self):
        # Returns the symbolic expression of the model
        expression = 0
        for k in range(len(self.terms_list)):
            meijer_gk, _, w_k = self.terms_list[k]
            argument_str = "[ReLU(P" + str(k + 1) + ")]"
            argument_symbol = Symbol(argument_str)
            expression += w_k * meijer_gk.expression(x=argument_symbol)
        return expression


    def get_projections(self):
        # Returns the projections appearing in the symbolic expression
        proj_list = []
        for k in range(len(self.terms_list)):
            _, vs, _ = self.terms_list[k]
            vks = [vs[i*self.dim_x[0]: (i+1)*self.dim_x[0]] for i in range(self.nclasses)]
            offset = self.nclasses*self.dim_x[0]
            uks = [vs[offset + i*self.dim_x[1]: offset + (i+1)*self.dim_x[1]] for i in range(self.nclasses)]
            class_proj_list = []

            for vk, uk in zip(vks, uks): 
                symbol_k = 0
                for jj in range(self.dim_x[0]):
                    for ii in range(self.dim_x[1]):
                        symbol_k += vk[jj] * Symbol("X" + str(j + 1)) * uk[ii]
                class_proj_list.append(symbol_k)
            proj_list.append(class_proj_list)
        return proj_list


    def string_projections(self):
        # Returns a string containing all projections
        proj_strs = []
        proj_list = self.get_projections()
        for ck in range(len(proj_list)):
            proj_str = ""
            for k in range(len(proj_list[ck])):
                proj_str += "P" + str(k + 1) + " = " + str(proj_list[ck][k]) + "\n"
            proj_strs.append(proj_str)
        return proj_strs


    def print_projections(self):
        # Prints the projections appearing in the symbolic expression
        proj_list = self.get_projections()
        for ck in range(len(proj_list)):
            for k in range(len(proj_list[ck])):
                print ("expression for class {}".format(k)) 
                print("P" + str(k + 1) + " = ", proj_list[ck][k])



    def get_feature_importance(self, x0):
        # Returns the feature importance for a prediction at x0
        importance_list = np.zeros(self.dim_x)
        lambda_ = np.max(np.linalg.eig(x0)[0])

        for k in range(len(self.terms_list)):
            g_k, v_k, w_k = self.terms_list[k]

            v_ks = [v_k[i*self.dim_x[0]: (i+1)*self.dim_x[0]] for i in range(self.nclasses)]
            offset = self.nclasses*self.dim_x[0]
            u_ks = [v_k[offset + i*self.dim_x[1]: offset + (i+1)*self.dim_x[1]] for i in range(self.nclasses)]


            x_ks = np.array([self.forward(x0, v_k, u_k) for v_k, u_k in zip(v_ks, u_ks)])

            v_k = v_ks[np.argmax(x_ks)]
            u_k = u_ks[np.argmax(x_ks)]
            x_k = max(self.eps, np.argmax(x_ks))

            if x_k > 0:
                coef_k = mpmath.chop(mpmath.taylor(g_k.math_expr, x_k, 1))
                for ii in range(self.dim_x[0]):
                    for jj in range(self.dim_x[1]):
                        importance_list[ii, jj] += sympify(
                            w_k * coef_k[1] * v_k[ii] * u_k[jj]/ (lambda_ * np.linalg.norm(u_k) * np.linalg.norm(v_k)))

        return importance_list


        # Change the model:

    def tune_new_term(self, X, g_order, theta_0):
        # Tunes a new term for the model for f with a Meijer G-function of order g_order

        _, _, p, q = g_order

        def split_theta(theta):
            # Splits theta in the Meijer G-function part, the vector part and the weight part
            theta_g = np.concatenate((theta[:p + q], np.array([1.0])))
            theta_vs = theta[p + q:p + q + self.nclasses*self.dim_x[0]]
            theta_us = theta[p + q + self.nclasses*self.dim_x[0]:-1]
            theta_w = theta[-1]
            return theta_g, theta_vs, theta_us, theta_w

        def loss(theta):
            # Computes the loss for a new term of parameter theta
            residual_list = self.current_resi
            theta_g, vs_, us_, w_ = split_theta(theta)

            vs_ = [vs_[i*self.dim_x[0]: (i+1)*self.dim_x[0]] for i in range(self.nclasses)]
            us_ = [us_[i*self.dim_x[1]: (i+1)*self.dim_x[1]] for i in range(self.nclasses)]
            meijer_g_ = MeijerG(theta=theta_g, order=g_order)

            Ys = []

            for ii in range(self.nclasses):
                Y = w_ * meijer_g_.evaluate(self.forward(X, vs_[ii], us_[ii]))
                Ys.append(Y)

            Ys = softmax(np.array(Ys).T, 1)

            # loss_ = np.mean((Y - residual_list) ** 2)
            loss_ = np.mean(self.residual(residual_list, np.array(Ys), 'NLL'))
            # print("Loss: ", loss_)
            return loss_

        # perpendicularity conditions
        index = np.arange(self.nclasses)
        cmb_idxs = list(combinations(index, 2))
        contrainMatrix = np.zeros((len(cmb_idxs), len(theta_0)))

        for ii, cmb_idx in enumerate(cmb_idxs):
            contrainMatrix[ii][p + q + cmb_idx[0]*self.dim_x[0] : p + q + (cmb_idx[0] + 1)*self.dim_x[0]] = \
                                theta_0[p + q + cmb_idx[1]*self.dim_x[0] : p + q + (cmb_idx[1] + 1)*self.dim_x[0]] / \
                                (np.linalg.norm(theta_0[p + q + cmb_idx[1]*self.dim_x[0] : p + q + (cmb_idx[1] + 1)*self.dim_x[0]]) *\
                                    np.linalg.norm(theta_0[p + q + cmb_idx[0]*self.dim_x[0] : p + q + (cmb_idx[0] + 1)*self.dim_x[0]]))

            offset = p + q + self.nclasses*self.dim_x[0] 
            contrainMatrix[ii][offset + cmb_idx[0]*self.dim_x[1] : offset + (cmb_idx[0] + 1)*self.dim_x[1]] = \
                        theta_0[offset + cmb_idx[1]*self.dim_x[1] : offset + (cmb_idx[1] + 1)*self.dim_x[1]] / \
                            (np.linalg.norm(theta_0[offset + cmb_idx[1]*self.dim_x[1] : offset + (cmb_idx[1] + 1)*self.dim_x[1]]) *\
                            np.linalg.norm(theta_0[offset + cmb_idx[0]*self.dim_x[1] : offset + (cmb_idx[0] + 1)*self.dim_x[1]]))


        # FIXME 
        lefteq = righteq = np.zeros(len(cmb_idxs))
        linear_constraint = LinearConstraint(contrainMatrix, lefteq, righteq)

        new_theta, new_loss = self.optimize_CG(loss, theta_0, linear_constraint)
        new_theta_meijer, new_vs, new_us, new_w = split_theta(new_theta)
        new_meijerg = MeijerG(theta=new_theta_meijer, order=g_order)
        return new_meijerg, new_vs, new_us, new_w, new_loss


    def residual(self, true, pred, type_='L1'):
        # Default: true - pred
        if type_ == 'L1':
            return np.abs(true - self.alpha*pred)

        elif type_ == 'L2':
            return np.abs(true - pred)**2

        #NLL
        elif type_ == 'CE':
            rtrue = np.zeros_like(true)
            rtrue[:, np.argmax(true, 1)] = 1.0
            res = -1.*np.sum(rtrue*np.log(pred + EPS), axis=1)
            return res

        elif type_ == 'NLL':
            # inspireed by actor critic
            # true is residual element similar to advantage term in AC3
            # effective loss would be -log_prob*advantage
            
            log_prob = np.log(pred + EPS)
            log_prob = -1.*np.sum(log_prob*true, axis=1)
            return log_prob


    def fit(self, f, X):
        # Fits a model for f via a projection pursuit strategy
        self.n_points = len(X)
        self.dim_x = X[0].shape
        h_dic = load_h()
        loss_tol = self.loss_tol

        w0 = 1.0
        count = 0
        Y_target = f(X[..., None])
        current_loss = 10000.0
        
        self.loss_list.append(current_loss)
        while current_loss > loss_tol:
            count += 1
            new_loss_list = []
            new_terms_list = []

            np.random.seed(self.random_seed)

            v0s = np.array([np.random.randn(self.dim_x[0]) for _ in range(self.nclasses)])
            u0s = np.array([np.random.randn(self.dim_x[1]) for _ in range(self.nclasses)])

            self.current_resi = self.residual(Y_target, self.predict(X), 'L1')

            if self.verbosity:
                print(100 * "%")
                print("Now working on term number ", count, ".")
            

            for k in range(len(h_dic)):
                if self.verbosity:
                    print(100 * "=")
                    print("Now working on hyperparameter tree number ", k + 1, ".")
            
                theta_g0, g_order = h_dic['hyper_' + str(k + 1)]
                v0 = v0s.reshape(-1)
                u0 = u0s.reshape(-1)

                theta_0 = np.concatenate((theta_g0, v0, u0, [w0]))

                new_meijer_g, new_v, new_u, new_w, new_loss = self.tune_new_term(X, g_order, theta_0)
                new_loss_list.append(new_loss)
                new_terms_list.append([new_meijer_g, new_v, new_u, new_w])

                if new_loss < loss_tol:
                    print(100 * "=")
                    print("The algorithm stopped because the desired precision was achieved.")
                    break

            best_index = np.argmin(np.array(new_loss_list))
            best_term = new_terms_list[int(best_index)]
            best_loss = new_loss_list[int(best_index)]

            if best_loss / current_loss < self.ratio_tol:
                self.terms_list.append(best_term)
                self.loss_list.append(best_loss)
                if self.verbosity:
                    print(100 * "=")
                    print("The tree number ", best_index + 1, " was selected as the best.")
                self.backfit(f, X[..., None])
                current_loss = self.loss_list[-1]
            else:
                print(100 * "=")
                print("The algorithm stopped because it was unable to find a term"
                      " that significantly decreases the loss.")
                break

            if self.verbosity:
                print(100 * "=")
                print(100 * "=")
                print("The current model has the following expression: ", self)
                print("The current value of the loss is: ", current_loss, ".")

        print(100 * "-")
        print(100 * "-")
        print("The final model has the following expression:")
        print(self)
        self.print_projections()
        print("The number of terms inside the expansion is", len(self.terms_list), ".")
        print("The current loss is", self.loss_list[-1], ".")
        print(100 * "-")


    def backfit(self, f, X):
        # The backfitting procedure invoked at each iteration of fit to correct the previous terms
        for k in range(len(self.terms_list) - 1):
            if self.verbosity:
                print(100 * "=")
                print("Now backfitting term number ", k + 1, ".")
            self.current_resi = self.residual(f(X), self.predict(X, exclude_term=True, exclusion_id=k), 'L1')
            meijer_g0, v0, w0 = self.terms_list[k]
            theta_meijer0 = meijer_g0.theta[:-1]

            theta0 = np.concatenate((theta_meijer0, v0, [w0]))
            g_order = meijer_g0.order
            new_meijerg, new_v, new_w, new_loss = self.tune_new_term(X, g_order, theta0)
            if new_loss < self.loss_list[-1]:
                self.terms_list[k] = [new_meijerg, new_v, new_w]
                self.loss_list[-1] = new_loss
        if self.verbosity:
            print(100 * "=")
            print("Backfitting complete.")
