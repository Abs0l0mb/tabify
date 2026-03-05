'use strict';

import {
    Block,
    Menu,
    MenuItem,
    MenuItemData,
    ClientLocation,
    Button,
    Api,
    Tools
} from '@src/classes';

export class UserMenuItem extends MenuItem {

    constructor(data: MenuItemData, menu: Menu, parent: Block) {

        super(data, menu, parent);
    
        this.drawLogoutButton();
    }

    /*
    **
    **
    */
    private drawLogoutButton() : void {
        
        const logoutButton = new Button({
            label: ''
        }, this).onNative('click', async (event: Event) => {

            event.stopPropagation();

            logoutButton.load();

            try {
                await Api.post('/logout');
                ClientLocation.get().api.checkAuth();
                await Tools.sleep(500);
                this.menu.emit('item-click');
                logoutButton.unload();
            } catch(error) {
                logoutButton.unload();
                ClientLocation.get().api.checkAuth();
            }
        });
    }
}