import sqlite3
# Nasdaq100 테이블을 생성하기 위한 SQL문을 정의한다.
stmts = '''
    DROP TABLE IF EXISTS Nasdaq100;
    CREATE TABLE Nasdaq100(
    순번 INTEGER PRIMARY KEY,
    날짜 TEXT,
    시가 REAL,
    고가 REAL,
    저가 REAL,
    종가 REAL,
    수정종가 REAL,
    거래량 INTEGER,
    종목명 TEXT
    );
    '''
# 데이터베이스에 연결한다.
conn = sqlite3.connect('stock.db')

try:
    # 커서를 생성한다.
    cursor = conn.cursor()
    
    # 커서로 Nasdaq100 테이블을 생성한다.
    cursor.executescript(stmts)
    
    # 트랜잭션을 커밋한다.(변경사항을 저장)
    conn.commit()
except sqlite3.Error as err: # 에러 발생시 에러 메세지 출력
    print('SQLite ERROR:', err)
finally:  
    conn.close()  # 데이터베이스 사용 후 연결을 해제한다.
    print('Job completed!')

    import csv

# 데이터베이스에 연결한다.
conn = sqlite3.connect('stock.db')

try:
    cursor = conn.cursor() # 커서를 생성한다.

    csv_reader = csv.reader(open('NASDAQ_100_Data_From_2010.csv'), delimiter='\t') # 탭을 구분자로 하여 NASDAQ_100_Data_From_2010.csv를 읽을 csv_reader 생성
    next(csv_reader) # 헤더 제거

    # CSV 데이터를 테이블로 입력한다.
    for row in csv_reader:
    # CSV 각 행의 데이터를 Nasdaq100 테이블에 삽입
        cursor.execute('''INSERT INTO Nasdaq100 (날짜, 시가, 고가, 저가, 종가, 수정종가, 거래량, 종목명) VALUES(?, ?, ?, ?, ?, ?, ?, ?);''', row)

    # 트랜잭션을 커밋한다.
    conn.commit()
    
except sqlite3.Error as err:
    print('SQLite ERROR:', err)
finally:
    print('DONE!')
    conn.close() # 데이터베이스 사용 후 연결을 해제한다.