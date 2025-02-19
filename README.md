![](ball.jpeg)  

Predykcja ALL NBA i Rookie Team

📌 Opis działania kodu

Kod rozpoczyna się od linii 247, gdzie wczytywane są dane ze statystykami od lat 2000-2023, wyniki ALL NBA oraz statystyki Rookie.

Następnie dane są filtrowane:

🛑 Usuwani są gracze z liczbą mniejszą niż 65 rozegranych meczów.

📏 Cechy są skalowane.

Kod zawiera dwie funkcje odpowiedzialne za predykcje dla ALL NBA i Rookie. Argument year pozwala na sprawdzenie danych dla konkretnego roku. Obie funkcje działają na tej samej zasadzie, jednak filtracja danych przebiega w nich inaczej.

🏀 ALL NBA - Opis działania

🔍 1. Przetwarzanie danych:

📂 Filtracja danych oraz łączenie statystyk z wynikami ALL NBA.

🔢 Graczom przypisywane są wartości:

0 - jeśli nie należeli do żadnego zespołu,

1 - dla trzeciego zespołu,

2 - dla drugiego zespołu,

3 - dla pierwszego zespołu.

🗑️ Usuwany jest pierwszy rok danych (2000).

📊 2. Analiza korelacji:

Dane trafiają do funkcji correlation_cal, która tworzy korelacje między cechami a wynikami.

🎯 Wartość value określa próg dla cech, które zostaną zwrócone.

🎭 3. Filtracja graczy:

📉 Funkcja filter_top_players_by_year filtruje 40% najgorszych graczy na podstawie cech z korelacji.

🏆 Dane są podzielone na graczy w pozycjach F, C oraz G (uwzględniając również none).

🔄 Dane ponownie przechodzą przez correlation_cal, aby wybrać cechy mające największy wpływ.

🏗 4. Tworzenie zestawów danych:

📚 Dane z wybranymi cechami trafiają do funkcji make_collection, która rozdziela dane na zestawy testowe i treningowe.

🤖 5. Budowanie modelu:

🧠 Modele zostały dobrane na podstawie ich wpływu na dane oraz reakcji na nie.

🔄 Dane są przekazywane do różnych modeli w funkcji evaluate, która zwraca listy zawodników.

🏅 6. Ostateczna selekcja zawodników:

🔍 Funkcja remove_duplicate_players usuwa powtarzających się zawodników w różnych zespołach z mniejszą liczbą wystąpień.

📜 Ostatecznie zwracana jest lista zawierająca 5 najczęściej występujących zawodników z każdego zespołu.

🌟 All_NBA_Rookie_Team

Funkcja działa analogicznie do predykcji ALL NBA, z kilkoma różnicami:

⚖️ Dobór wag jest inny.

🚫 Brak podziału na pozycje ze względu na niepełne dane.

📌 Podsumowanie

Kod umożliwia analizę i predykcję wyników dla zespołów ALL NBA i Rookie Team, bazując na statystykach graczy oraz korelacjach między cechami a wynikami. Umożliwia dostosowanie analizy do konkretnego roku oraz automatyczną selekcję najlepszych zawodników.

