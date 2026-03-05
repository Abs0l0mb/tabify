'use strict';

import {
    Page,
    LoginPopup,
    Popup,
    Div,
    ClientLocation
} from '@src/classes';

export class LoginPage extends Page {
    
    static popup: Popup;

    constructor() {

        super('Login');

        if (LoginPage.popup)
            LoginPage.popup.hide();

        LoginPage.popup = new LoginPopup();
    }
}