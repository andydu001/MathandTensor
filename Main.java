package org.example;

import fr.cnes.sirius.patrius.math.stat.descriptive.moment.StandardDeviation;
import fr.cnes.sirius.patrius.math.stat.descriptive.rank.Max;
import fr.cnes.sirius.patrius.math.stat.descriptive.rank.Min;
import fr.cnes.sirius.patrius.math.stat.descriptive.summary.Product;
import fr.cnes.sirius.patrius.math.stat.descriptive.summary.SumOfLogs;
import fr.cnes.sirius.patrius.math.stat.regression.SimpleRegression;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;

import java.util.Arrays;
import java.util.function.BiConsumer;


public class Main {
    public static void main(String[] args) {

        FloatNdArray matrix = NdArrays.ofFloats(Shape.of(2,3,2));
        matrix.set(NdArrays.vectorOf(6.0f, 1.0f), 0, 1)
                .set(NdArrays.vectorOf(3.0f, 6.0f), 0, 2)
                .set(NdArrays.vectorOf(8.0f, 2.0f), 1, 0)
                .set(NdArrays.vectorOf(9.0f, 10.0f), 1, 1)
                .set(NdArrays.vectorOf(1.0f, 2.0f), 1, 2);


        matrix.scalars().forEachIndexed(new BiConsumer<long[], FloatNdArray>() {
            @Override
            public void accept(long[] longs, FloatNdArray floatNdArray) {
                System.out.println(Arrays.toString(longs) + ": " + floatNdArray);
            }
        });


        SimpleRegression r = new SimpleRegression();
        double[][] x = {{2, 2,6}, {2, 2, 3, 4}, {2, 2, 3}, {2, 2, 3}, {2, 2, 3}, {2,2}, {2,2,4}, {2,4,5}};
        r.addData(x);
        //  r.addData(x);


        System.out.println(r.regress().hasIntercept());
        System.out.println(Arrays.toString(r.regress().getParameterEstimates()));
        System.out.println(r.regress().getNumberOfParameters());
        System.out.println(r.getTotalSumSquares());



        Max max = new Max();
        // max.evaluate();
        double[] bl = {2,3,4,4,5,5,6,7,123};

        Min min = new Min();

        System.out.println(min.evaluate(bl));
        System.out.println(max.evaluate(bl));

        // calculate the standard deviation
        StandardDeviation standardDeviation = new StandardDeviation();
        double sd = standardDeviation.evaluate(bl);
        System.out.println(sd);

        // calculate the sum of logs
        SumOfLogs sumOfLogs = new SumOfLogs();
        double sum = sumOfLogs.evaluate(bl);
        System.out.println(sum);

        // calc
        Product product = new Product();
        double pr = product.evaluate(bl);
        System.out.println(pr);

        // DataType.forNumber(DataType.DT_DOUBLE_VALUE).;

        //  Tensor tensor = Tensor.of(TFloating.class,Shape.of(2,3,2));

    }
}