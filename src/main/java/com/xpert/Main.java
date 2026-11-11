
package com.xpert;

public class Main {

    public static void main(String[] args) {

        if (args.length != 2) {
            System.err.println(
                    "Usage: ./run.sh <input_dir> <output.json>"
            );
            System.exit(1);
        }

        String inputDirectory = args[0];
        String outputFile = args[1];

        System.out.println("Xpert Cold Start Pipeline");
        System.out.println("Input: " + inputDirectory);
        System.out.println("Output: " + outputFile);
    }
}
