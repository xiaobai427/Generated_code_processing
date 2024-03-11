void DRIVER_KUNLUN::set_radio_init()
{
	reg_write(0x0167, 0x08); // refpll init
	reg_write(0x0256, 0x00); // tx init
	Sleep(1 us); // delay 1us for settling
	reg_write(0x04e6, 0x19); // rx init
	reg_write(0x19, 0x11); // rx_pmu_on
	Sleep(1 us); // delay 1us for settling
}


void DRIVER_KUNLUN::tx0_ldo_on(uint8_t enable)
{
	if (enable == 0)
	{
		reg_write(0x01f9, 0x0); // disable LDO
	}
	else if (enable == 1)
	{
		reg_write(0x01f9, 0x1c); // enable LDO
	}
	Sleep(1 us); // delay 1us for settling
}


void DRIVER_KUNLUN::set_rx_tpana(uint8_t tp_setting)
{
	reg_write(0x013B, 0x01); // enable RX_TPANA
	reg_write(0x0141, 0x01); // enable RX_TPANA
	reg_write(0x013C, tp_setting); // select TPMUX
}


void DRIVER_KUNLUN::set_rx0_tia_vcm(uint8_t tia_vcm, uint8_t mixer_bias)
{
	reg_write(0x007C, ((tia_vcm<<4)&0xf0)+mixer_bias); // merge and assign code
	Sleep(0.1 us); // delay 0.1us for settling
}


void DRIVER_KUNLUN::rx0_on(uint8_t enable)
{
	rx0_bias0_on();
	rx_bias1_en(enable);
	rx_bias2_en();
}

void DRIVER_KUNLUN::rx0_bias0_on()
{
	reg_write(0x0070, 0x01); //
}

void DRIVER_KUNLUN::rx_bias1_en(uint8_t enable)
{
	reg_write(0x0071, enable); //
}

void DRIVER_KUNLUN::rx_bias2_en()
{
	reg_write(0x0072, 0x01); //
	reg_write(0x0073, 0x01); //
	reg_write(0x0074, 0x01); //
}


void DRIVER_KUNLUN::set_radio_cal()
{
	set_radio_cal_pll("GL_OTP_FPLL_PFDLDO15_VOSEL_ATE");
	set_radio_cal_lo("GL_OTP_LODIST_MLDO08_VRCAL_ATE");
	set_radio_cal_rx("GL_OTP_RX_TIA_MX_VCM_L0_ATE");
	set_radio_cal_adc("GL_OTP_ADC_LDO08_VOSEL_ATE");
}

void DRIVER_KUNLUN::set_radio_cal_pll(string trim_value)
{
	reg_write(0x018e, trim_value); //
}

void DRIVER_KUNLUN::set_radio_cal_pll(uint8_t trim_value)
{
	reg_write(0x018e, trim_value); //
}

void DRIVER_KUNLUN::set_radio_cal_lo(string trim_value)
{
	reg_write(0x01d7, trim_value); //
}

void DRIVER_KUNLUN::set_radio_cal_lo(uint8_t trim_value)
{
	reg_write(0x01d7, trim_value); //
}

void DRIVER_KUNLUN::set_radio_cal_rx(string trim_value)
{
	reg_write(0x007c, trim_value); //
	reg_write(0x00a7, trim_value); //
	reg_write(0x04be, trim_value); //
	reg_write(0x04e9, trim_value); //
}

void DRIVER_KUNLUN::set_radio_cal_rx(uint8_t trim_value)
{
	reg_write(0x007c, trim_value); //
	reg_write(0x00a7, trim_value); //
	reg_write(0x04be, trim_value); //
	reg_write(0x04e9, trim_value); //
}

void DRIVER_KUNLUN::set_radio_cal_adc(string trim_value)
{
	reg_write(0xD4, trim_value); //
}

void DRIVER_KUNLUN::set_radio_cal_adc(uint8_t trim_value)
{
	reg_write(0xD4, trim_value); //
}
void DRIVER_KUNLUN::set_radio_cal_cbc(uint8_t trim_value)
{
	reg_write(0x1d, trim_value); //
}

void DRIVER_KUNLUN::set_radio_cal_cbc(string trim_value)
{
	reg_write(0x1d, trim_value); //
}


void DRIVER_KUNLUN::set_radio_cal_tx(uint8_t trim_value)
{
	reg_write(0x01eb, trim_value); //
}

void DRIVER_KUNLUN::set_radio_cal_tx(string trim_value)
{
	reg_write(0x01eb, trim_value); //
}



void DRIVER_KUNLUN::set_radio_cal_cbc()
{
	set_radio_cal_cbc("GL_OTP_RXBG_CTAT_8U_CAL_ATE");
}

void DRIVER_KUNLUN::set_radio_cal_tx()
{
	set_radio_cal_tx("GL_OTP_TXLO_13G_LDO08_VRCAL_ATE");
}

void DRIVER_KUNLUN::set_GL_OTP_ALL()
{
	rdi.runTimeVal("GL_OTP_ADC_LDO08_VOSEL_ATE",GL_OTP_ADC_LDO08_VOSEL_ATE);
	rdi.runTimeVal("GL_OTP_FPLL_PFDLDO15_VOSEL_ATE",GL_OTP_FPLL_PFDLDO15_VOSEL_ATE);
	rdi.runTimeVal("GL_OTP_LODIST_MLDO08_VRCAL_ATE",GL_OTP_LODIST_MLDO08_VRCAL_ATE);
	rdi.runTimeVal("GL_OTP_RXBG_CTAT_8U_CAL_ATE",GL_OTP_RXBG_CTAT_8U_CAL_ATE);
	rdi.runTimeVal("GL_OTP_RX_TIA_MX_VCM_L0_ATE",GL_OTP_RX_TIA_MX_VCM_L0_ATE);
	rdi.runTimeVal("GL_OTP_TXLO_13G_LDO08_VRCAL_ATE",GL_OTP_TXLO_13G_LDO08_VRCAL_ATE);
}

void (DRIVER_KUNLUN::*functionPointers[400])(uint8_t) =
{
	&DRIVER_KUNLUN::set_driver_index, // this line is only to take index-0;
	&DRIVER_KUNLUN::set_radio_cal_pll,
	&DRIVER_KUNLUN::set_radio_cal_lo,
	&DRIVER_KUNLUN::set_radio_cal_rx,
	&DRIVER_KUNLUN::set_radio_cal_adc,
	&DRIVER_KUNLUN::set_radio_cal_cbc,
	&DRIVER_KUNLUN::set_radio_cal_tx

};



