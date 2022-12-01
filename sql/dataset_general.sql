/*
 Navicat Premium Data Transfer

 Source Server         : [Localhost] Mysql Xampp
 Source Server Type    : MySQL
 Source Server Version : 100136
 Source Host           : localhost:3306
 Source Schema         : datasets

 Target Server Type    : MySQL
 Target Server Version : 100136
 File Encoding         : 65001

 Date: 07/05/2019 15:01:36
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for bund
-- ----------------------------
DROP TABLE IF EXISTS `bund`;
CREATE TABLE `sp500_cet`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `date` date NOT NULL,
  `time` time(0) NOT NULL,
  `open` float NOT NULL,
  `close` float NOT NULL,
  `close_adj` float NOT NULL,
  `high` float NOT NULL,
  `low` float NOT NULL,
  `up` int(11) NOT NULL,
  `down` int(11) NOT NULL,
  `volume` float NOT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1201668 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Compact;

SET FOREIGN_KEY_CHECKS = 1;
