
/****** Object:  Database [Test_FI_Portfolio]    Script Date: 30-04-2025 14:41:23 ******/
--CREATE DATABASE [Test_Portfolio]
--GO

USE Test_FI_Portfolio
GO

CREATE TABLE [dbo].[FixedIncomeSecurities](
	[SecurityId] [int] IDENTITY(1,1) NOT NULL,
	[ISIN] [nvarchar](20) NOT NULL,
	[Ticker] [nvarchar](50) NULL,
	[Issuer] [nvarchar](255) NULL,
	[CouponRate] [decimal](5, 2) NULL,
	[MaturityDate] [date] NULL,
	[CurrencyCode] [char](3) NULL,
	[CreatedDate] [datetime] NULL,
PRIMARY KEY CLUSTERED 
(
	[SecurityId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY],
UNIQUE NONCLUSTERED 
(
	[ISIN] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[OCIChanges]    Script Date: 30-04-2025 14:41:23 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[OCIChanges](
	[OCIChangeId] [int] IDENTITY(1,1) NOT NULL,
	[HoldingId] [int] NOT NULL,
	[ChangeDate] [date] NOT NULL,
	[UnrealizedGainLoss] [decimal](18, 4) NOT NULL,
	[Notes] [nvarchar](500) NULL,
	[CreatedDate] [datetime] NULL,
PRIMARY KEY CLUSTERED 
(
	[OCIChangeId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PortfolioHoldings]    Script Date: 30-04-2025 14:41:23 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PortfolioHoldings](
	[HoldingId] [int] IDENTITY(1,1) NOT NULL,
	[PortfolioId] [int] NOT NULL,
	[SecurityId] [int] NOT NULL,
	[PurchaseDate] [date] NULL,
	[Quantity] [decimal](18, 4) NULL,
	[PurchasePrice] [decimal](18, 4) NULL,
	[AmortizedCost] [decimal](18, 4) NULL,
	[CreatedDate] [datetime] NULL,
PRIMARY KEY CLUSTERED 
(
	[HoldingId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Portfolios]    Script Date: 30-04-2025 14:41:23 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Portfolios](
	[PortfolioId] [int] IDENTITY(1,1) NOT NULL,
	[PortfolioName] [nvarchar](255) NOT NULL,
	[Description] [nvarchar](500) NULL,
	[CreatedDate] [datetime] NULL,
PRIMARY KEY CLUSTERED 
(
	[PortfolioId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Valuations]    Script Date: 30-04-2025 14:41:23 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Valuations](
	[ValuationId] [int] IDENTITY(1,1) NOT NULL,
	[HoldingId] [int] NOT NULL,
	[ValuationDate] [date] NOT NULL,
	[MarketValue] [decimal](18, 4) NOT NULL,
	[FairValueAdjustment] [decimal](18, 4) NULL,
	[CreatedDate] [datetime] NULL,
PRIMARY KEY CLUSTERED 
(
	[ValuationId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO
ALTER TABLE [dbo].[FixedIncomeSecurities] ADD  DEFAULT (getdate()) FOR [CreatedDate]
GO
ALTER TABLE [dbo].[OCIChanges] ADD  DEFAULT (getdate()) FOR [CreatedDate]
GO
ALTER TABLE [dbo].[PortfolioHoldings] ADD  DEFAULT (getdate()) FOR [CreatedDate]
GO
ALTER TABLE [dbo].[Portfolios] ADD  DEFAULT (getdate()) FOR [CreatedDate]
GO
ALTER TABLE [dbo].[Valuations] ADD  DEFAULT (getdate()) FOR [CreatedDate]
GO
ALTER TABLE [dbo].[OCIChanges]  WITH CHECK ADD FOREIGN KEY([HoldingId])
REFERENCES [dbo].[PortfolioHoldings] ([HoldingId])
GO
ALTER TABLE [dbo].[PortfolioHoldings]  WITH CHECK ADD FOREIGN KEY([PortfolioId])
REFERENCES [dbo].[Portfolios] ([PortfolioId])
GO
ALTER TABLE [dbo].[PortfolioHoldings]  WITH CHECK ADD FOREIGN KEY([SecurityId])
REFERENCES [dbo].[FixedIncomeSecurities] ([SecurityId])
GO
ALTER TABLE [dbo].[Valuations]  WITH CHECK ADD FOREIGN KEY([HoldingId])
REFERENCES [dbo].[PortfolioHoldings] ([HoldingId])
GO


SET IDENTITY_INSERT [dbo].[Portfolios] ON 
GO
INSERT [dbo].[Portfolios] ([PortfolioId], [PortfolioName], [Description], [CreatedDate]) VALUES (1, N'Corporate Bonds Portfolio', N'Investment Grade Bonds', CAST(N'2025-04-26T11:38:06.153' AS DateTime))
GO
INSERT [dbo].[Portfolios] ([PortfolioId], [PortfolioName], [Description], [CreatedDate]) VALUES (2, N'Sovereign Bonds Portfolio', N'Government Securities Portfolio', CAST(N'2025-04-26T11:38:06.153' AS DateTime))
GO
SET IDENTITY_INSERT [dbo].[Portfolios] OFF
GO
SET IDENTITY_INSERT [dbo].[FixedIncomeSecurities] ON 
GO
INSERT [dbo].[FixedIncomeSecurities] ([SecurityId], [ISIN], [Ticker], [Issuer], [CouponRate], [MaturityDate], [CurrencyCode], [CreatedDate]) VALUES (1, N'US1234567890', N'USCORP1', N'US Corp 1', CAST(5.25 AS Decimal(5, 2)), CAST(N'2030-12-31' AS Date), N'USD', CAST(N'2025-04-26T11:38:06.210' AS DateTime))
GO
INSERT [dbo].[FixedIncomeSecurities] ([SecurityId], [ISIN], [Ticker], [Issuer], [CouponRate], [MaturityDate], [CurrencyCode], [CreatedDate]) VALUES (2, N'US2234567890', N'USCORP2', N'US Corp 2', CAST(4.75 AS Decimal(5, 2)), CAST(N'2029-06-30' AS Date), N'USD', CAST(N'2025-04-26T11:38:06.210' AS DateTime))
GO
INSERT [dbo].[FixedIncomeSecurities] ([SecurityId], [ISIN], [Ticker], [Issuer], [CouponRate], [MaturityDate], [CurrencyCode], [CreatedDate]) VALUES (3, N'IN1234567890', N'INBOND1', N'India Govt', CAST(6.50 AS Decimal(5, 2)), CAST(N'2035-03-31' AS Date), N'INR', CAST(N'2025-04-26T11:38:06.210' AS DateTime))
GO
INSERT [dbo].[FixedIncomeSecurities] ([SecurityId], [ISIN], [Ticker], [Issuer], [CouponRate], [MaturityDate], [CurrencyCode], [CreatedDate]) VALUES (4, N'IN2234567890', N'INBOND2', N'India Govt', CAST(7.00 AS Decimal(5, 2)), CAST(N'2038-09-30' AS Date), N'INR', CAST(N'2025-04-26T11:38:06.210' AS DateTime))
GO
INSERT [dbo].[FixedIncomeSecurities] ([SecurityId], [ISIN], [Ticker], [Issuer], [CouponRate], [MaturityDate], [CurrencyCode], [CreatedDate]) VALUES (5, N'GB1234567890', N'UKBOND1', N'UK Govt', CAST(3.50 AS Decimal(5, 2)), CAST(N'2031-01-31' AS Date), N'GBP', CAST(N'2025-04-26T11:38:06.210' AS DateTime))
GO
INSERT [dbo].[FixedIncomeSecurities] ([SecurityId], [ISIN], [Ticker], [Issuer], [CouponRate], [MaturityDate], [CurrencyCode], [CreatedDate]) VALUES (6, N'JP1234567890', N'JPBOND1', N'Japan Govt', CAST(0.50 AS Decimal(5, 2)), CAST(N'2032-04-30' AS Date), N'JPY', CAST(N'2025-04-26T11:38:06.210' AS DateTime))
GO
INSERT [dbo].[FixedIncomeSecurities] ([SecurityId], [ISIN], [Ticker], [Issuer], [CouponRate], [MaturityDate], [CurrencyCode], [CreatedDate]) VALUES (7, N'US3234567890', N'USCORP3', N'US Corp 3', CAST(5.00 AS Decimal(5, 2)), CAST(N'2028-11-30' AS Date), N'USD', CAST(N'2025-04-26T11:38:06.210' AS DateTime))
GO
INSERT [dbo].[FixedIncomeSecurities] ([SecurityId], [ISIN], [Ticker], [Issuer], [CouponRate], [MaturityDate], [CurrencyCode], [CreatedDate]) VALUES (8, N'FR1234567890', N'FRBOND1', N'France Govt', CAST(1.25 AS Decimal(5, 2)), CAST(N'2033-07-15' AS Date), N'EUR', CAST(N'2025-04-26T11:38:06.210' AS DateTime))
GO
INSERT [dbo].[FixedIncomeSecurities] ([SecurityId], [ISIN], [Ticker], [Issuer], [CouponRate], [MaturityDate], [CurrencyCode], [CreatedDate]) VALUES (9, N'AU1234567890', N'AUBOND1', N'Australia Govt', CAST(2.75 AS Decimal(5, 2)), CAST(N'2034-05-15' AS Date), N'AUD', CAST(N'2025-04-26T11:38:06.210' AS DateTime))
GO
INSERT [dbo].[FixedIncomeSecurities] ([SecurityId], [ISIN], [Ticker], [Issuer], [CouponRate], [MaturityDate], [CurrencyCode], [CreatedDate]) VALUES (10, N'DE1234567890', N'DEBOND1', N'Germany Govt', CAST(0.90 AS Decimal(5, 2)), CAST(N'2030-10-15' AS Date), N'EUR', CAST(N'2025-04-26T11:38:06.210' AS DateTime))
GO
SET IDENTITY_INSERT [dbo].[FixedIncomeSecurities] OFF
GO
SET IDENTITY_INSERT [dbo].[PortfolioHoldings] ON 
GO
INSERT [dbo].[PortfolioHoldings] ([HoldingId], [PortfolioId], [SecurityId], [PurchaseDate], [Quantity], [PurchasePrice], [AmortizedCost], [CreatedDate]) VALUES (1, 1, 1, CAST(N'2022-01-15' AS Date), CAST(100000.0000 AS Decimal(18, 4)), CAST(102.5000 AS Decimal(18, 4)), CAST(101.0000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.273' AS DateTime))
GO
INSERT [dbo].[PortfolioHoldings] ([HoldingId], [PortfolioId], [SecurityId], [PurchaseDate], [Quantity], [PurchasePrice], [AmortizedCost], [CreatedDate]) VALUES (2, 1, 2, CAST(N'2022-03-20' AS Date), CAST(50000.0000 AS Decimal(18, 4)), CAST(101.0000 AS Decimal(18, 4)), CAST(100.0000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.273' AS DateTime))
GO
INSERT [dbo].[PortfolioHoldings] ([HoldingId], [PortfolioId], [SecurityId], [PurchaseDate], [Quantity], [PurchasePrice], [AmortizedCost], [CreatedDate]) VALUES (3, 1, 7, CAST(N'2023-02-10' AS Date), CAST(80000.0000 AS Decimal(18, 4)), CAST(99.0000 AS Decimal(18, 4)), CAST(99.5000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.273' AS DateTime))
GO
INSERT [dbo].[PortfolioHoldings] ([HoldingId], [PortfolioId], [SecurityId], [PurchaseDate], [Quantity], [PurchasePrice], [AmortizedCost], [CreatedDate]) VALUES (4, 2, 3, CAST(N'2022-05-10' AS Date), CAST(120000.0000 AS Decimal(18, 4)), CAST(98.7500 AS Decimal(18, 4)), CAST(99.0000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.273' AS DateTime))
GO
INSERT [dbo].[PortfolioHoldings] ([HoldingId], [PortfolioId], [SecurityId], [PurchaseDate], [Quantity], [PurchasePrice], [AmortizedCost], [CreatedDate]) VALUES (5, 2, 4, CAST(N'2022-08-15' AS Date), CAST(100000.0000 AS Decimal(18, 4)), CAST(100.5000 AS Decimal(18, 4)), CAST(100.0000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.273' AS DateTime))
GO
INSERT [dbo].[PortfolioHoldings] ([HoldingId], [PortfolioId], [SecurityId], [PurchaseDate], [Quantity], [PurchasePrice], [AmortizedCost], [CreatedDate]) VALUES (6, 2, 5, CAST(N'2021-12-10' AS Date), CAST(70000.0000 AS Decimal(18, 4)), CAST(101.2500 AS Decimal(18, 4)), CAST(100.0000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.273' AS DateTime))
GO
INSERT [dbo].[PortfolioHoldings] ([HoldingId], [PortfolioId], [SecurityId], [PurchaseDate], [Quantity], [PurchasePrice], [AmortizedCost], [CreatedDate]) VALUES (7, 2, 6, CAST(N'2021-09-25' AS Date), CAST(90000.0000 AS Decimal(18, 4)), CAST(100.0000 AS Decimal(18, 4)), CAST(99.7500 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.273' AS DateTime))
GO
INSERT [dbo].[PortfolioHoldings] ([HoldingId], [PortfolioId], [SecurityId], [PurchaseDate], [Quantity], [PurchasePrice], [AmortizedCost], [CreatedDate]) VALUES (8, 1, 8, CAST(N'2022-04-20' AS Date), CAST(60000.0000 AS Decimal(18, 4)), CAST(100.7500 AS Decimal(18, 4)), CAST(100.0000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.273' AS DateTime))
GO
INSERT [dbo].[PortfolioHoldings] ([HoldingId], [PortfolioId], [SecurityId], [PurchaseDate], [Quantity], [PurchasePrice], [AmortizedCost], [CreatedDate]) VALUES (9, 2, 9, CAST(N'2023-01-15' AS Date), CAST(50000.0000 AS Decimal(18, 4)), CAST(99.5000 AS Decimal(18, 4)), CAST(99.0000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.273' AS DateTime))
GO
INSERT [dbo].[PortfolioHoldings] ([HoldingId], [PortfolioId], [SecurityId], [PurchaseDate], [Quantity], [PurchasePrice], [AmortizedCost], [CreatedDate]) VALUES (10, 1, 10, CAST(N'2022-07-10' AS Date), CAST(75000.0000 AS Decimal(18, 4)), CAST(98.0000 AS Decimal(18, 4)), CAST(98.5000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.273' AS DateTime))
GO
SET IDENTITY_INSERT [dbo].[PortfolioHoldings] OFF
GO
SET IDENTITY_INSERT [dbo].[Valuations] ON 
GO
INSERT [dbo].[Valuations] ([ValuationId], [HoldingId], [ValuationDate], [MarketValue], [FairValueAdjustment], [CreatedDate]) VALUES (1, 1, CAST(N'2025-04-25' AS Date), CAST(10300000.0000 AS Decimal(18, 4)), CAST(7500.0000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.303' AS DateTime))
GO
INSERT [dbo].[Valuations] ([ValuationId], [HoldingId], [ValuationDate], [MarketValue], [FairValueAdjustment], [CreatedDate]) VALUES (2, 2, CAST(N'2025-04-25' AS Date), CAST(5050000.0000 AS Decimal(18, 4)), CAST(5000.0000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.303' AS DateTime))
GO
INSERT [dbo].[Valuations] ([ValuationId], [HoldingId], [ValuationDate], [MarketValue], [FairValueAdjustment], [CreatedDate]) VALUES (3, 3, CAST(N'2025-04-25' AS Date), CAST(7920000.0000 AS Decimal(18, 4)), CAST(3200.0000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.303' AS DateTime))
GO
INSERT [dbo].[Valuations] ([ValuationId], [HoldingId], [ValuationDate], [MarketValue], [FairValueAdjustment], [CreatedDate]) VALUES (4, 4, CAST(N'2025-04-25' AS Date), CAST(11850000.0000 AS Decimal(18, 4)), CAST(4500.0000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.303' AS DateTime))
GO
INSERT [dbo].[Valuations] ([ValuationId], [HoldingId], [ValuationDate], [MarketValue], [FairValueAdjustment], [CreatedDate]) VALUES (5, 5, CAST(N'2025-04-25' AS Date), CAST(10100000.0000 AS Decimal(18, 4)), CAST(5000.0000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.303' AS DateTime))
GO
INSERT [dbo].[Valuations] ([ValuationId], [HoldingId], [ValuationDate], [MarketValue], [FairValueAdjustment], [CreatedDate]) VALUES (6, 6, CAST(N'2025-04-25' AS Date), CAST(7070000.0000 AS Decimal(18, 4)), CAST(7000.0000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.303' AS DateTime))
GO
INSERT [dbo].[Valuations] ([ValuationId], [HoldingId], [ValuationDate], [MarketValue], [FairValueAdjustment], [CreatedDate]) VALUES (7, 7, CAST(N'2025-04-25' AS Date), CAST(8950000.0000 AS Decimal(18, 4)), CAST(4000.0000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.303' AS DateTime))
GO
INSERT [dbo].[Valuations] ([ValuationId], [HoldingId], [ValuationDate], [MarketValue], [FairValueAdjustment], [CreatedDate]) VALUES (8, 8, CAST(N'2025-04-25' AS Date), CAST(6045000.0000 AS Decimal(18, 4)), CAST(4500.0000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.303' AS DateTime))
GO
INSERT [dbo].[Valuations] ([ValuationId], [HoldingId], [ValuationDate], [MarketValue], [FairValueAdjustment], [CreatedDate]) VALUES (9, 9, CAST(N'2025-04-25' AS Date), CAST(4975000.0000 AS Decimal(18, 4)), CAST(3500.0000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.303' AS DateTime))
GO
INSERT [dbo].[Valuations] ([ValuationId], [HoldingId], [ValuationDate], [MarketValue], [FairValueAdjustment], [CreatedDate]) VALUES (10, 10, CAST(N'2025-04-25' AS Date), CAST(7400000.0000 AS Decimal(18, 4)), CAST(4000.0000 AS Decimal(18, 4)), CAST(N'2025-04-26T11:38:06.303' AS DateTime))
GO
SET IDENTITY_INSERT [dbo].[Valuations] OFF
GO
SET IDENTITY_INSERT [dbo].[OCIChanges] ON 
GO
INSERT [dbo].[OCIChanges] ([OCIChangeId], [HoldingId], [ChangeDate], [UnrealizedGainLoss], [Notes], [CreatedDate]) VALUES (1, 1, CAST(N'2025-04-25' AS Date), CAST(7500.0000 AS Decimal(18, 4)), N'Positive valuation adjustment', CAST(N'2025-04-26T11:38:06.333' AS DateTime))
GO
INSERT [dbo].[OCIChanges] ([OCIChangeId], [HoldingId], [ChangeDate], [UnrealizedGainLoss], [Notes], [CreatedDate]) VALUES (2, 2, CAST(N'2025-04-25' AS Date), CAST(5000.0000 AS Decimal(18, 4)), N'Positive valuation adjustment', CAST(N'2025-04-26T11:38:06.333' AS DateTime))
GO
INSERT [dbo].[OCIChanges] ([OCIChangeId], [HoldingId], [ChangeDate], [UnrealizedGainLoss], [Notes], [CreatedDate]) VALUES (3, 3, CAST(N'2025-04-25' AS Date), CAST(3200.0000 AS Decimal(18, 4)), N'Positive valuation adjustment', CAST(N'2025-04-26T11:38:06.333' AS DateTime))
GO
INSERT [dbo].[OCIChanges] ([OCIChangeId], [HoldingId], [ChangeDate], [UnrealizedGainLoss], [Notes], [CreatedDate]) VALUES (4, 4, CAST(N'2025-04-25' AS Date), CAST(4500.0000 AS Decimal(18, 4)), N'Positive valuation adjustment', CAST(N'2025-04-26T11:38:06.333' AS DateTime))
GO
INSERT [dbo].[OCIChanges] ([OCIChangeId], [HoldingId], [ChangeDate], [UnrealizedGainLoss], [Notes], [CreatedDate]) VALUES (5, 5, CAST(N'2025-04-25' AS Date), CAST(5000.0000 AS Decimal(18, 4)), N'Positive valuation adjustment', CAST(N'2025-04-26T11:38:06.333' AS DateTime))
GO
INSERT [dbo].[OCIChanges] ([OCIChangeId], [HoldingId], [ChangeDate], [UnrealizedGainLoss], [Notes], [CreatedDate]) VALUES (6, 6, CAST(N'2025-04-25' AS Date), CAST(7000.0000 AS Decimal(18, 4)), N'Positive valuation adjustment', CAST(N'2025-04-26T11:38:06.333' AS DateTime))
GO
INSERT [dbo].[OCIChanges] ([OCIChangeId], [HoldingId], [ChangeDate], [UnrealizedGainLoss], [Notes], [CreatedDate]) VALUES (7, 7, CAST(N'2025-04-25' AS Date), CAST(4000.0000 AS Decimal(18, 4)), N'Positive valuation adjustment', CAST(N'2025-04-26T11:38:06.333' AS DateTime))
GO
INSERT [dbo].[OCIChanges] ([OCIChangeId], [HoldingId], [ChangeDate], [UnrealizedGainLoss], [Notes], [CreatedDate]) VALUES (8, 8, CAST(N'2025-04-25' AS Date), CAST(4500.0000 AS Decimal(18, 4)), N'Positive valuation adjustment', CAST(N'2025-04-26T11:38:06.333' AS DateTime))
GO
INSERT [dbo].[OCIChanges] ([OCIChangeId], [HoldingId], [ChangeDate], [UnrealizedGainLoss], [Notes], [CreatedDate]) VALUES (9, 9, CAST(N'2025-04-25' AS Date), CAST(3500.0000 AS Decimal(18, 4)), N'Positive valuation adjustment', CAST(N'2025-04-26T11:38:06.333' AS DateTime))
GO
INSERT [dbo].[OCIChanges] ([OCIChangeId], [HoldingId], [ChangeDate], [UnrealizedGainLoss], [Notes], [CreatedDate]) VALUES (10, 10, CAST(N'2025-04-25' AS Date), CAST(4000.0000 AS Decimal(18, 4)), N'Positive valuation adjustment', CAST(N'2025-04-26T11:38:06.333' AS DateTime))
GO
SET IDENTITY_INSERT [dbo].[OCIChanges] OFF
GO


