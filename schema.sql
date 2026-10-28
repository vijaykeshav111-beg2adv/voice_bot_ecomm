-- Appointment table for the voice bot (MySQL 5.7+ / 8.x / MariaDB 10.3+).
-- db.py runs this automatically on first use, after creating the database if it is missing.
-- To do it by hand instead, run in the mysql client:
--   CREATE DATABASE voicebot CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
--   USE voicebot
-- and then execute this file.
--
-- Double-booking is prevented by the database itself: "active" is 1 while an appointment is
-- booked and NULL once it is cancelled. MySQL allows many NULLs in a unique index, so a
-- cancelled slot can be booked again, but two active bookings of one slot are impossible,
-- even if two callers try at the same instant.

CREATE TABLE IF NOT EXISTS appointments (
  id            INT UNSIGNED NOT NULL AUTO_INCREMENT,
  ref_code      CHAR(6)      NOT NULL,
  customer_name VARCHAR(100) NOT NULL,
  purpose       VARCHAR(255) NULL,
  slot_start    DATETIME     NOT NULL,
  slot_end      DATETIME     NOT NULL,
  status        VARCHAR(10)  NOT NULL DEFAULT 'booked',
  active        TINYINT      NULL     DEFAULT 1,
  channel       VARCHAR(10)  NOT NULL DEFAULT 'web',
  created_at    DATETIME     NOT NULL,
  updated_at    DATETIME     NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_ref_code (ref_code),
  UNIQUE KEY uq_active_slot (slot_start, active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
