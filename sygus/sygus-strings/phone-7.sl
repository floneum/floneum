(set-logic SLIA)
 
(synth-fun f ((name String)) String
    ((Start String (ntString))
     (ntString String (name " " "+" "-" "."
                       (str.++ ntString ntString)
                       (str.replace ntString ntString ntString)
                       (str.at ntString ntInt)
                       (int.to.str ntInt)
                       (str.substr ntString ntInt ntInt)))
      (ntInt Int (0 1 2 3 4 5
                  (+ ntInt ntInt)
                  (- ntInt ntInt)
                  (str.len ntString)
                  (str.to.int ntString)
                  (str.indexof ntString ntString ntInt)))
      (ntBool Bool (true false
                    (str.prefixof ntString ntString)
                    (str.suffixof ntString ntString)
                    (str.contains ntString ntString)))))


(declare-var name String)

(constraint (= (f "+106 769-858-438") "858"))
(constraint (= (f "+83 973-757-831") "757"))
(constraint (= (f "+62 647-787-775") "787"))
(constraint (= (f "+172 027-507-632") "507"))
(constraint (= (f "+72 001-050-856") "050"))
(constraint (= (f "+95 310-537-401") "537"))
(constraint (= (f "+6 775-969-238") "969"))

(check-synth)
