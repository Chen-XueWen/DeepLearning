classdef HelperFunc
    methods(Static)
        
        function SigZ = sigmoid(z)
            SigZ = 1./(1+exp(-z));
        end
        
        function [w,b] = initializeweight(dim)
            w = zeros(dim, 1);
            b = zeros(1);
        end
        
        function [dw,db,J] = propagate(w,b,X,Y)
            %Forward Propagation
            [Dim,NumSets] = size(X);
            m = NumSets;
            A = HelperFunc.sigmoid(w.'*X + b);
            J = (-1/m)*sum(Y.*log(A)+(1-Y).*log(1-A));
            dw = (1/m)*X*(A-Y).';
            db = (1/m)*sum((A-Y));
        end
        
        function [w,b,dw,db,J] = optimize(w, b, X, Y, iterations, learning_rate)
            for Run = 1:iterations
                [dw,db,J] = HelperFunc.propagate(w,b,X,Y);
                Cost(Run,1) = J;
                w = w - learning_rate*dw;
                b = b - learning_rate*db;
                if (rem(Run,100) == 0)
                    disp("Cost after iteration " + Run + ": " + J)
                end
            end
            plot(Cost);
            xlabel("Iterations");
            ylabel("Cost");
            title("Iterations vs Cost")
        end
        
        function Y_Prediction = predict(w,b,X)
            [Dim,NumSets] = size(X);
            Y_Prediction = zeros(1,NumSets);
            Activation = HelperFunc.sigmoid(w.'*X + b);
            Sets = length(Activation);
            for i = 1:Sets
                if Activation(i) > 0.5
                    Y_Prediction(i) = 1;
                else
                    Y_Prediction(i) = 0;
                end
            end
        end
        
        
    end
end